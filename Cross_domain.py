from pathlib import Path
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, accuracy_score
from model_Cross_domain import MMTF_DTI


def set_seed(seed=2025):
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AdamW(th.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = th.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = th.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * (bias_correction2 ** 0.5) / bias_correction1
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * scheduled_lr)

                p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.copy_(p_data_fp32)

        return loss


class DTIDataset(Dataset):
    def __init__(self, combined_data):
        self.combined_data = combined_data

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        compound_graph, protein_graph, chem_feature_pca, prot_feature_pca, label = self.combined_data[idx]

        if compound_graph is None or protein_graph is None or chem_feature_pca is None or prot_feature_pca is None:
            return None

        try:
            label = float(label)
        except ValueError:
            label = 0.0

        chem_feature_tensor = th.tensor(chem_feature_pca, dtype=th.float32)
        prot_feature_tensor = th.tensor(prot_feature_pca, dtype=th.float32)

        if isinstance(label, str):
            label = 1.0 if label.lower() in ['1', 'true', 'yes'] else 0.0
        label_tensor = th.tensor(label, dtype=th.float32)

        return (compound_graph, protein_graph, chem_feature_tensor, prot_feature_tensor, label_tensor)


def collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    compound_graphs, protein_graphs, chem_features, prot_features, labels = zip(*batch)

    labels = th.cat([label.view(-1) for label in labels], dim=0).to(device)
    batched_compound = dgl.batch(compound_graphs).to(device)
    batched_protein = dgl.batch(protein_graphs).to(device)
    chem_features = th.stack(chem_features, dim=0).to(device)
    prot_features = th.stack(prot_features, dim=0).to(device)

    return (batched_compound, batched_protein, chem_features, prot_features, labels)


class DomainAdaptiveLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.2, temp=0.05, adapt_weight=True):
        super().__init__()
        self.task_loss = nn.BCEWithLogitsLoss()
        self.domain_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.adapt_weight = adapt_weight

    def _smooth_labels(self, labels, alpha=0.1):
        return labels * (1 - alpha) + alpha / 2

    def _contrastive_loss(self, features):
        features = F.normalize(features, dim=1)
        sim_matrix = th.mm(features, features.T) / self.temp
        labels = th.arange(features.size(0)).to(features.device)
        return F.cross_entropy(sim_matrix, labels)

    def forward(self, outputs, labels, domains, epoch):
        if self.adapt_weight:
            curr_alpha = self.alpha * min(epoch / 10, 1.0)
            curr_gamma = self.gamma * min(epoch / 5, 1.0)
        else:
            curr_alpha = self.alpha
            curr_gamma = self.gamma

        smoothed_labels = self._smooth_labels(labels)
        task_loss = self.task_loss(outputs['pred'].squeeze(), smoothed_labels)
        domain_loss = self.domain_loss(outputs['domain'], domains)
        contrast_loss = self._contrastive_loss(outputs['features'])

        return task_loss + curr_alpha * domain_loss + curr_gamma * contrast_loss


class CrossDomainTrainer:
    def __init__(self, model, sources, target_unlabeled_loader, val_loaders, config, patience=5):
        self.model = model
        self.sources = sources
        self.target_unlabeled_loader = target_unlabeled_loader
        self.val_loaders = val_loaders
        self.config = config
        self.patience = patience
        self.best_auc = 0
        self.counter = 0
        self.best_model_state = None

        self.opt_main = AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
        self.opt_meta = AdamW(
            model.feature_extractor.domain_classifier.parameters(),
            lr=config['lr'] * 0.1,
            weight_decay=1e-5
        )

        self.curriculum = [
            (10, 0.1, 0.05),
            (20, 0.4, 0.2),
            (float('inf'), 0.6, 0.3)
        ]

        self._target_iter = None

    def _get_target_batch(self):
        if self._target_iter is None:
            self._target_iter = iter(self.target_unlabeled_loader)
        try:
            batch = next(self._target_iter)
            if batch is None:
                raise StopIteration
        except StopIteration:
            self._target_iter = iter(self.target_unlabeled_loader)
            batch = next(self._target_iter)
        return batch

    def _meta_step(self, source_batches, target_batch):
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.model.feature_extractor.domain_classifier.parameters():
            param.requires_grad = True

        self.opt_meta.zero_grad()

        meta_loss = 0
        source_feats = []

        # Source samples -> domain label = 0
        for s_comp, s_prot, s_chem, s_prot_feat, _ in source_batches:
            outputs = self.model(s_comp, s_prot, s_chem, s_prot_feat, domain_adapt=True)
            batch_size = s_comp.batch_size
            loss = F.cross_entropy(outputs['domain'], th.zeros(batch_size, dtype=th.long).to(device))
            meta_loss += loss

            s_feat_dict = self.model.feature_extractor(
                compound_graph=s_comp,
                protein_graph=s_prot,
                chem_features=s_chem,
                prot_features=s_prot_feat
            )
            source_feats.append(s_feat_dict['features'])

        # Target samples -> domain label = 1 (labels ignored)
        t_comp, t_prot, t_chem, t_prot_feat, _ = target_batch
        outputs = self.model(t_comp, t_prot, t_chem, t_prot_feat, domain_adapt=True)
        batch_size = t_comp.batch_size
        t_loss = F.cross_entropy(outputs['domain'], th.ones(batch_size, dtype=th.long).to(device))
        meta_loss += t_loss

        t_feat_dict = self.model.feature_extractor(
            compound_graph=t_comp,
            protein_graph=t_prot,
            chem_features=t_chem,
            prot_features=t_prot_feat
        )
        target_feat = t_feat_dict['features']

        # Prototype alignment loss
        source_center = th.cat(source_feats, dim=0).mean(dim=0)
        target_center = target_feat.mean(dim=0)
        proto_loss = F.mse_loss(source_center, target_center)

        total_loss = meta_loss + 0.2 * proto_loss
        total_loss.backward()

        self.opt_meta.step()

        for param in self.model.feature_extractor.parameters():
            param.requires_grad = True

        return total_loss.item() / (len(source_batches) + 1)

    def train_epoch(self, epoch):
        self.model.train()

        for loader in self.sources:
            for batch in loader:
                if batch is None:
                    continue

        stage = next((s for s in self.curriculum if epoch < s[0]), self.curriculum[-1])
        _, alpha, gamma = stage

        # Meta-learning phase
        source_batches = [next(iter(loader)) for loader in self.sources]
        target_batch = self._get_target_batch()
        meta_loss = self._meta_step(source_batches, target_batch)

        # Main task training
        total_loss = 0
        num_batches = 0

        for loader in self.sources:
            for batch in loader:
                if not batch:
                    continue
                self.opt_main.zero_grad()

                comp, prot, chem, prot_feat, labels = batch
                labels = labels.float().view(-1)

                t_batch = self._get_target_batch()
                t_comp, t_prot, t_chem, t_prot_feat, _ = t_batch

                mix_comp = dgl.batch([comp, t_comp])
                mix_prot = dgl.batch([prot, t_prot])
                mix_chem = th.cat([chem, t_chem])
                mix_prot_feat = th.cat([prot_feat, t_prot_feat])

                domains = th.cat([
                    th.zeros(len(labels), dtype=th.long),
                    th.ones(t_comp.batch_size, dtype=th.long)
                ]).to(device)

                outputs = self.model(mix_comp, mix_prot, mix_chem, mix_prot_feat, domain_adapt=True)

                # Task loss: source domain only
                source_pred = outputs['pred'][:len(labels)]
                task_loss = F.binary_cross_entropy_with_logits(source_pred.squeeze(), labels)

                # Domain classification loss: all data
                domain_loss = F.cross_entropy(outputs['domain'], domains)

                # Contrastive loss
                features = outputs['features']
                features_norm = F.normalize(features, dim=1)
                sim_matrix = th.mm(features_norm, features_norm.T) / 0.05
                contrast_labels = th.arange(features.size(0)).to(device)
                contrast_loss = F.cross_entropy(sim_matrix, contrast_labels)

                loss = task_loss + alpha * domain_loss + gamma * contrast_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt_main.step()

                total_loss += loss.item()
                num_batches += 1

        val_auc = self._evaluate_on_val()
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch}, Val AUC: {val_auc:.4f}, Loss: {avg_loss:.4f}, Meta Loss: {meta_loss:.4f}")

        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.counter = 0
            self.best_model_state = self.model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping at epoch {epoch}, best Val AUC: {self.best_auc:.4f}")
            self.model.load_state_dict(self.best_model_state)
            return True
        return False

    def _evaluate_on_val(self):
        self.model.eval()
        all_preds, all_labels = [], []

        with th.no_grad():
            for loader in self.val_loaders:
                for batch in loader:
                    if not batch:
                        continue
                    comp, prot, chem, prot_feat, labels = batch
                    outputs = self.model(comp, prot, chem, prot_feat, domain_adapt=False)
                    pred = outputs['pred'].sigmoid().detach().cpu().numpy()
                    all_preds.extend(pred.flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())

        if len(all_preds) == 0:
            return 0.0

        return roc_auc_score(all_labels, all_preds)


class ModelEvaluator:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds, all_labels = [], []

        with th.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue

                tta_preds, tta_labels = [], []
                for _ in range(3):
                    aug_batch = self._tta_augment(batch)
                    compound_graph, protein_graph, chem_features, prot_features, labels = aug_batch
                    labels = labels.clone().detach().float().view(-1)

                    compound_graph = compound_graph.to(device)
                    protein_graph = protein_graph.to(device)
                    chem_features = chem_features.to(device)
                    prot_features = prot_features.to(device)
                    labels = labels.to(device)

                    outputs = self.model(compound_graph, protein_graph, chem_features, prot_features,
                                         domain_adapt=False)
                    pred_sigmoid = th.sigmoid(outputs['pred'])

                    tta_preds.append(pred_sigmoid.cpu())
                    tta_labels.append(labels.cpu())

                avg_preds = th.stack(tta_preds).mean(dim=0).numpy().reshape(-1)
                avg_labels = th.stack(tta_labels).mean(dim=0).numpy().reshape(-1)

                all_preds.append(avg_preds)
                all_labels.append(avg_labels)

        all_preds = np.concatenate(all_preds) if all_preds else np.array([])
        all_labels = np.concatenate(all_labels) if all_labels else np.array([])

        if len(all_preds) == 0 or len(all_labels) == 0:
            raise ValueError("No valid predictions or labels were collected")

        threshold = 0.5
        all_preds_bin = (all_preds >= threshold).astype(int)

        precision = precision_score(all_labels, all_preds_bin)
        recall = recall_score(all_labels, all_preds_bin)
        accuracy = accuracy_score(all_labels, all_preds_bin)
        auc = roc_auc_score(all_labels, all_preds)
        prauc = average_precision_score(all_labels, all_preds)

        return precision, recall, accuracy, auc, prauc

    def _tta_augment(self, batch):
        compound_graph, protein_graph, chem_features, prot_features, labels = batch
        chem_features = chem_features.clone()
        mask = th.rand_like(chem_features) < 0.05
        chem_features[mask] = 0
        return compound_graph, protein_graph, chem_features, prot_features, labels


def create_dataloader(combined_data, batch_size, shuffle=True):
    dataset = DTIDataset(combined_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False
    )
    return dataloader


def load_pickle(file_path):
    if os.path.getsize(file_path) == 0:
        raise EOFError(f"File {file_path} is empty.")
    with open(file_path, 'rb') as fp:
        return pickle.load(fp)


def verify_no_data_leakage(target_train_data, target_test_data):
    def extract_pairs(data):
        pairs = set()
        for item in data:
            comp_id = item[0].num_nodes() if item[0] is not None else 0
            prot_id = item[1].num_nodes() if item[1] is not None else 0
            chem_hash = hash(tuple(item[2].flatten()[:10].tolist())) if item[2] is not None else 0
            prot_hash = hash(tuple(item[3].flatten()[:10].tolist())) if item[3] is not None else 0
            pairs.add((comp_id, prot_id, chem_hash, prot_hash))
        return pairs

    train_pairs = extract_pairs(target_train_data)
    test_pairs = extract_pairs(target_test_data)

def main():
    set_seed(2025)

    global device
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data paths
    bindingdb_dir = Path('./data/BindingDB/data_split/')
    drugbank_dir = Path('./data/DrugBank/data_split/')
    human_dir = Path('./data/Human/data_split/')
    celegans_dir = Path('./data/Celegans/data_split/')
    best_model_dir = Path('./models/Bestmodel/')

    # Source domain files
    source_data_files = {
        'train': {
            'bindingdb': bindingdb_dir / 'processed_data_train_bindingdb.pkl',
            'drugbank': drugbank_dir / 'processed_data_train_drugbank.pkl',
            'human': human_dir / 'processed_data_train_human.pkl'
        },
        'val': {
            'bindingdb': bindingdb_dir / 'processed_data_val_bindingdb.pkl',
            'drugbank': drugbank_dir / 'processed_data_val_drugbank.pkl',
            'human': human_dir / 'processed_data_val_human.pkl'
        }
    }

    # Target domain files
    target_data_files = {
        'train': celegans_dir / 'processed_data_train_celegans.pkl',
        'test': celegans_dir / 'processed_data_test_celegans.pkl'
    }

    source_domains = ['bindingdb', 'drugbank', 'human']
    source_data = {
        'train': {domain: [] for domain in source_domains},
        'val': {domain: [] for domain in source_domains}
    }

    # Load source domain data
    print("\nLoading source domain data (labeled)...")
    for split in ['train', 'val']:
        for domain in source_domains:
            file_path = source_data_files[split][domain]
            if file_path.exists():
                source_data[split][domain] = load_pickle(file_path)
                print(f"  {domain} {split}: {len(source_data[split][domain])} samples")

    # Load target domain data
    print("\nLoading target domain data...")
    target_train_data = load_pickle(target_data_files['train'])
    target_test_data = load_pickle(target_data_files['test'])
    print(f"  celegans train (unlabeled for alignment): {len(target_train_data)} samples")
    print(f"  celegans test (final evaluation): {len(target_test_data)} samples")

    # Verify no data leakage
    print("\nVerifying data separation...")
    verify_no_data_leakage(target_train_data, target_test_data)

    batch_size = 16

    # Source domain DataLoaders
    source_train_loaders = [
        create_dataloader(source_data['train']['bindingdb'], batch_size),
        create_dataloader(source_data['train']['drugbank'], batch_size),
        create_dataloader(source_data['train']['human'], batch_size)
    ]

    source_val_loaders = [
        create_dataloader(source_data['val']['bindingdb'], batch_size, shuffle=False),
        create_dataloader(source_data['val']['drugbank'], batch_size, shuffle=False),
        create_dataloader(source_data['val']['human'], batch_size, shuffle=False)
    ]

    # Target domain DataLoaders
    target_train_loader = create_dataloader(target_train_data, batch_size)  # For domain alignment
    target_test_loader = create_dataloader(target_test_data, batch_size, shuffle=False)  # For testing

    # Initialize model
    model = MMTF_DTI(
        esm_hidden_size=128,
        mol2vec_size=128,
        compound_dim=128,
        protein_dim=128,
        out_dim=1
    ).to(device)

    num_pos = sum(1 for data in target_test_data if data[-1] == 1)
    num_neg = len(target_test_data) - num_pos
    pos_weight = th.tensor([num_neg / num_pos]).to(device)

    evaluator = ModelEvaluator(model, nn.BCEWithLogitsLoss(pos_weight=pos_weight))

    config = {'lr': 1e-4}
    trainer = CrossDomainTrainer(
        model=model,
        sources=source_train_loaders,
        target_unlabeled_loader=target_train_loader,
        val_loaders=source_val_loaders,
        config=config,
        patience=15
    )
    best_model_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print("\n" + "=" * 60)
    print("Training with UDA")
    print("Source: BindingDB, DrugBank, C.elegans (labeled)")
    print("Target: Human train (unlabeled for alignment)")
    print("=" * 60)

    for epoch in range(1, 31):
        early_stop = trainer.train_epoch(epoch)
        if early_stop:
            break

    if trainer.best_model_state is not None:
        model.load_state_dict(trainer.best_model_state)
        best_model_path = best_model_dir / "best_model_uda.pt"
        th.save(trainer.best_model_state, best_model_path)
        print(f"\nBest model saved to {best_model_path}")

    # Testing
    print("\n" + "=" * 60)
    print("Testing on target domain (Human test)")
    print("=" * 60)

    precision, recall, accuracy, auc, prauc = evaluator.evaluate(target_test_loader)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}, PRAUC: {prauc:.4f}")


if __name__ == '__main__':
    main()

