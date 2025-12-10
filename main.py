from pathlib import Path
import os
import pickle
import random
import gc
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
from model import MMTF_DTI
import torch.optim as optim
import traceback
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

DATASET_NAME = 'DrugBank'  # Options: 'DrugBank', 'BindingDB', 'Human', 'Celegans'
DATA_DIR = Path('./data/')
MODEL_DIR = Path('./models/')
OUTPUT_DIR = Path('./output/')


def set_seed(seed=2024):
    """Set random seed for reproducibility."""
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class CustomDataset(Dataset):
    def __init__(self, combined_data, augment=False):
        self.combined_data = combined_data
        self.augment = augment

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        compound_graph, protein_graph, chem_feature_pca, prot_feature_pca, label = self.combined_data[idx]

        if self.augment:
            compound_graph = self.augment_molecular_graph(compound_graph)

        if compound_graph is None or protein_graph is None or chem_feature_pca is None or prot_feature_pca is None:
            return None

        chem_feature_tensor = th.tensor(chem_feature_pca, dtype=th.float32)
        prot_feature_tensor = th.tensor(prot_feature_pca, dtype=th.float32)
        label_tensor = th.tensor(label, dtype=th.float32)

        return (compound_graph, protein_graph, chem_feature_tensor, prot_feature_tensor, label_tensor)

    def augment_molecular_graph(self, graph):
        num_nodes = graph.num_nodes()
        rotation_matrix = np.random.rand(3, 3)
        rotation_matrix, _ = np.linalg.qr(rotation_matrix)
        pos = graph.ndata['pos'].numpy()
        pos = np.dot(pos, rotation_matrix)
        graph.ndata['pos'] = th.tensor(pos, dtype=th.float32)
        return graph


def collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    compound_graphs, protein_graphs, chem_features, prot_features, labels = zip(*batch)

    batched_compound_graph = dgl.batch(compound_graphs).to(device)
    batched_protein_graph = dgl.batch(protein_graphs).to(device)

    if 'h' in batched_compound_graph.ndata:
        batched_compound_graph.ndata['h'] = batched_compound_graph.ndata['h'].float()
    else:
        raise KeyError("Compound graph does not contain node feature 'h'.")

    if 'h' in batched_protein_graph.ndata:
        batched_protein_graph.ndata['h'] = batched_protein_graph.ndata['h'].float()
    else:
        raise KeyError("Protein graph does not contain node feature 'h'.")

    chem_features = th.stack(chem_features, dim=0).to(device)
    prot_features = th.stack(prot_features, dim=0).to(device)
    labels = th.stack(labels, dim=0).to(device)

    return (batched_compound_graph, batched_protein_graph, chem_features, prot_features, labels)


class Trainer:
    def __init__(self, model, optimizer, criterion, gradient_accumulation):
        self.model = model
        self.gradient_accumulation = gradient_accumulation
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        self.optimizer.zero_grad()

        # 添加进度条
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', leave=False)
        for step, batch in pbar:
            if batch is None:
                continue
            compound_graph, protein_graph, chem_features, prot_features, labels = batch

            compound_graph = compound_graph.to(device)
            protein_graph = protein_graph.to(device)
            chem_features = chem_features.to(device)
            prot_features = prot_features.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = self.model(compound_graph, protein_graph, chem_features, prot_features)

            loss = self.criterion(outputs, labels)
            loss = loss / self.gradient_accumulation
            loss.backward()

            th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if (step + 1) % self.gradient_accumulation == 0 or (step + 1) == len(dataloader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            all_preds.append(outputs.sigmoid().cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

            pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation:.4f}'})

        avg_loss = total_loss / len(dataloader)
        return avg_loss, np.concatenate(all_labels), np.concatenate(all_preds)

    def save_model(self, path):
        path_str = str(path)
        parent_dir = os.path.dirname(path_str)

        if not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except Exception as e:
                print(f"Failed to create directory {parent_dir}: {e}")
                traceback.print_exc()
                return False

        if not os.path.isdir(parent_dir):
            print(f"Parent path {parent_dir} exists but is not a directory.")
            return False

        try:
            th.save(self.model.state_dict(), path_str)
            print(f"Model saved to {path_str}")
            return True
        except Exception as e:
            print(f"Failed to save model to {path_str}: {e}")
            traceback.print_exc()
            return False


class Tester:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def test_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        pbar = tqdm(dataloader, total=len(dataloader), desc='Validating', leave=False)
        with th.no_grad():
            for batch in pbar:
                if batch is None:
                    continue
                compound_graph, protein_graph, chem_features, prot_features, labels = batch

                compound_graph = compound_graph.to(device)
                protein_graph = protein_graph.to(device)
                chem_features = chem_features.to(device)
                prot_features = prot_features.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                outputs = self.model(compound_graph, protein_graph, chem_features, prot_features)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                all_preds.append(outputs.sigmoid().cpu().detach().numpy())
                all_labels.append(labels.cpu().detach().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        return avg_loss, np.concatenate(all_labels), np.concatenate(all_preds)


def create_dataloader(combined_data, batch_size, shuffle=True, augment=False):
    dataset = CustomDataset(combined_data, augment=augment)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=0
    )
    return dataloader


def augment_data(data):
    augmented_data = []
    for sample in data:
        compound_graph, protein_graph, chem_feature_pca, prot_feature_pca, label = sample
        noise_level = 0.2
        chem_feature_pca += np.random.normal(0, noise_level, size=chem_feature_pca.shape)
        prot_feature_pca += np.random.normal(0, noise_level, size=prot_feature_pca.shape)
        augmented_sample = (compound_graph, protein_graph, chem_feature_pca, prot_feature_pca, label)
        augmented_data.append(augmented_sample)
    return augmented_data


def load_pickle(file_path):
    if os.path.getsize(file_path) == 0:
        raise EOFError(f"File {file_path} is empty.")
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)


def main():
    set_seed(2024)
    global device
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup paths
    dataset_name_lower = DATASET_NAME.lower()
    data_dir = DATA_DIR / DATASET_NAME / 'data_split'
    best_model_dir = MODEL_DIR / 'Bestmodel'
    output_dir = OUTPUT_DIR

    # Create directories
    best_model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    processed_data_files = {
        'train': data_dir / f'processed_data_train_{dataset_name_lower}.pkl',
        'val': data_dir / f'processed_data_val_{dataset_name_lower}.pkl',
        'test': data_dir / f'processed_data_test_{dataset_name_lower}.pkl'
    }

    processed_data = {}
    for dataset_type, file_path in processed_data_files.items():
        print(f"Loading {dataset_type} dataset from {file_path}...")
        try:
            data = load_pickle(file_path)
            processed_data[dataset_type] = data
            print(f"Loaded {dataset_type} dataset with {len(data)} samples.")
        except Exception as e:
            print(f"Failed to load {dataset_type} dataset: {e}")
            traceback.print_exc()
            return

    # Data augmentation for training
    processed_data['train'] = augment_data(processed_data['train'])

    # Stratified K-fold cross-validation
    labels = [sample[-1] for sample in processed_data['train']]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    all_auc_scores = []
    epoch_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(processed_data['train'], labels)):
        print(f"\nFold {fold + 1}/5")
        train_data = [processed_data['train'][i] for i in train_idx]
        val_data = [processed_data['train'][i] for i in val_idx]

        batch_size = 128
        train_loader = create_dataloader(train_data, batch_size, shuffle=True, augment=False)
        val_loader = create_dataloader(val_data, batch_size, shuffle=False, augment=False)

        model = MMTF_DTI(
            esm_hidden_size=128,
            mol2vec_size=128,
            compound_dim=128,
            protein_dim=128,
            out_dim=1
        ).to(device)

        # Compute class weights for loss function
        all_train_labels = [sample[-1] for sample in train_data]
        all_train_labels_np = np.array(all_train_labels)
        num_pos = np.sum(all_train_labels_np == 1)
        num_neg = np.sum(all_train_labels_np == 0)
        pos_weight = th.tensor([num_neg / num_pos], dtype=th.float32).to(device)

        optimizer = th.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.5)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        trainer = Trainer(model, optimizer, criterion, gradient_accumulation=1)
        tester = Tester(model, criterion)

        best_prauc = 0
        patience = 20
        epochs_no_improve = 0

        for epoch in range(150):
            print(f"Epoch {epoch + 1}/150")
            train_loss, train_labels, train_preds = trainer.train_epoch(train_loader)
            val_loss, val_labels, val_preds = tester.test_epoch(val_loader)

            if len(val_labels) > 0 and len(val_preds) > 0:
                val_auc = roc_auc_score(val_labels, val_preds)
                val_prauc = average_precision_score(val_labels, val_preds)
                print(f"Validation AUC: {val_auc:.4f}, PRAUC: {val_prauc:.4f}")

                if val_prauc > best_prauc:
                    best_prauc = val_prauc
                    epochs_no_improve = 0
                    best_model_path = best_model_dir / f"best_model_fold_{fold + 1}.pt"
                    trainer.save_model(best_model_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break

                scheduler.step()
            else:
                print("No validation data available for metrics.")

            epoch_results.append({
                'fold': fold + 1,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_prauc': val_prauc
            })

        all_auc_scores.append(val_auc)

    print(f"\nAverage AUC over 5 folds: {np.mean(all_auc_scores):.4f}")

    results_df = pd.DataFrame(epoch_results)
    results_df.to_csv(output_dir / f'epoch_results_{dataset_name_lower}.csv', index=False)
    print(f"Epoch results saved to '{output_dir / f'epoch_results_{dataset_name_lower}.csv'}'.")

    # Testing
    print("\nTesting the best model...")
    test_loader = create_dataloader(processed_data['test'], batch_size=128, shuffle=False, augment=False)

    all_test_metrics = {
        'precision': [],
        'recall': [],
        'accuracy': [],
        'auc': [],
        'prauc': []
    }

    for fold in range(5):
        best_model_path = best_model_dir / f"best_model_fold_{fold + 1}.pt"
        if best_model_path.exists():
            model.load_state_dict(th.load(best_model_path))
            test_loss, test_labels, test_preds = tester.test_epoch(test_loader)

            if len(test_labels) > 0 and len(test_preds) > 0:
                test_preds_binary = (test_preds > 0.5).astype(int)

                precision = precision_score(test_labels, test_preds_binary)
                recall = recall_score(test_labels, test_preds_binary)
                accuracy = accuracy_score(test_labels, test_preds_binary)
                auc = roc_auc_score(test_labels, test_preds)
                prauc = average_precision_score(test_labels, test_preds)

                all_test_metrics['precision'].append(precision)
                all_test_metrics['recall'].append(recall)
                all_test_metrics['accuracy'].append(accuracy)
                all_test_metrics['auc'].append(auc)
                all_test_metrics['prauc'].append(prauc)

                print(f"Fold {fold + 1} Metrics: "
                      f"Precision: {precision:.4f}, "
                      f"Recall: {recall:.4f}, "
                      f"Accuracy: {accuracy:.4f}, "
                      f"AUC: {auc:.4f}, "
                      f"PRAUC: {prauc:.4f}")
            else:
                print(f"No test data available for metrics (Fold {fold + 1}).")
        else:
            print(f"Best model for Fold {fold + 1} not found.")

    if all_test_metrics['precision']:
        avg_precision = np.mean(all_test_metrics['precision'])
        avg_recall = np.mean(all_test_metrics['recall'])
        avg_accuracy = np.mean(all_test_metrics['accuracy'])
        avg_auc = np.mean(all_test_metrics['auc'])
        avg_prauc = np.mean(all_test_metrics['prauc'])

        print(f"\nAverage Test Metrics: "
              f"Precision: {avg_precision:.4f} ± {np.std(all_test_metrics['precision']):.4f}, "
              f"Recall: {avg_recall:.4f} ± {np.std(all_test_metrics['recall']):.4f}, "
              f"Accuracy: {avg_accuracy:.4f} ± {np.std(all_test_metrics['accuracy']):.4f}, "
              f"AUC: {avg_auc:.4f} ± {np.std(all_test_metrics['auc']):.4f}, "
              f"PRAUC: {avg_prauc:.4f} ± {np.std(all_test_metrics['prauc']):.4f}")
    else:
        print("No test metrics available.")


if __name__ == '__main__':
    main()
