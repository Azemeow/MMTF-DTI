import warnings

warnings.filterwarnings("ignore", message="please use MorganGenerator")

from rdkit import rdBase

rdBase.DisableLog('rdApp.warning')

import torch
import os
import pickle
import random
import numpy as np
import pandas as pd
from rdkit import Chem
import networkx as nx
from transformers import EsmModel, EsmTokenizer
import dgl
from mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import logging
from torch.utils.data import Dataset, DataLoader

DATASET_NAME = 'DrugBank'  # Options: 'DrugBank', 'BindingDB', 'Human', 'Celegans'
DATA_BASE_DIR = './data/'
MOL2VEC_PATH = 'model_300dim.pkl'
ESM_PATH = 'esm2_t12_35M_UR50D'
N_COMPONENTS_ESM = 128
N_COMPONENTS_MOL2VEC = 128

# Set random seed
random.seed(2024)
torch.manual_seed(2024)
np.random.seed(2024)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Mol2Vec Model
mol2vec_model_path = MOL2VEC_PATH
try:
    mol2vec_model = Word2Vec.load(mol2vec_model_path)
    print("Mol2Vec model loaded successfully.")
except Exception as e:
    print(f"Error loading Mol2Vec model: {e}")
    exit(1)

# Load ESM Model and Tokenizer
esm_tokenizer_path = ESM_PATH
try:
    esm_tokenizer = EsmTokenizer.from_pretrained(esm_tokenizer_path)
    print("ESM tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading ESM tokenizer: {e}")
    exit(1)

try:
    esm_model = EsmModel.from_pretrained(esm_tokenizer_path, use_safetensors=False)
    esm_model = esm_model.to(device)
    esm_model.eval()
    print("ESM model loaded successfully.")
except Exception as e:
    print(f"Error loading ESM model: {e}")
    exit(1)


def setup_logging(log_file='training.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def generate_mol2vec_feature(smile):
    try:
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {smile}")
        sentence = mol2alt_sentence(molecule, 1)
        mol2vec_feature = sentences2vec([sentence], mol2vec_model, unseen='UNK')[0]
    except Exception as e:
        print(f"Error generating Mol2Vec feature for {smile}: {e}")
        mol2vec_feature = np.zeros(mol2vec_model.vector_size, dtype=np.float32)

    if mol2vec_feature.shape[0] != mol2vec_model.vector_size:
        mol2vec_feature = np.resize(mol2vec_feature, mol2vec_model.vector_size)
    return mol2vec_feature


def generate_esm_protein_feature(sequence, esm_model, esm_tokenizer):
    try:
        inputs = esm_tokenizer(sequence, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            outputs = esm_model(**inputs)
        esm_features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        esm_feature = np.mean(esm_features, axis=0)
    except Exception as e:
        print(f"Error generating ESM feature: {e}")
        esm_feature = np.zeros(esm_model.config.hidden_size, dtype=np.float32)
    return esm_feature


def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smile}")

    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]

    node_features = np.array(
        [atom_features(atom, explicit_H=False, use_chirality=True) for atom in atoms],
        dtype=np.float32
    )

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    nx_graph = nx.from_numpy_array(adjacency)

    for i in range(n_atoms):
        nx_graph.nodes[i]['h'] = node_features[i]

    dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['h'])
    dgl_graph = dgl.add_self_loop(dgl_graph)

    return dgl_graph


def atom_features(atom, explicit_H=False, use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
         'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
         'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
         'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    ) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(
                  atom.GetHybridization(),
                  [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                   Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                   Chem.rdchem.HybridizationType.SP3D2]
              ) + [atom.GetIsAromatic()]

    if not explicit_H:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    if use_chirality:
        try:
            results += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + \
                       [atom.HasProp('_ChiralityPossible')]
        except:
            results += [False, False] + [atom.HasProp('_ChiralityPossible')]

    return results


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}.")
    return list(map(lambda s: x == s, allowable_set))


def protein_sequence_to_graph(sequence, feature=None, k=2):
    try:
        n = len(sequence)
        graph = nx.Graph()

        for i in range(n):
            if feature is not None:
                node_feature = feature
            else:
                node_feature = np.zeros(128, dtype=np.float32)
            graph.add_node(i, h=node_feature)

        for i in range(n):
            for j in range(i + 1, min(i + k + 1, n)):
                graph.add_edge(i, j)
                graph.add_edge(j, i)

        dgl_graph = dgl.from_networkx(graph, node_attrs=['h'])
        dgl_graph = dgl.add_self_loop(dgl_graph)

    except Exception as e:
        print(f"Error generating protein graph: {e}")
        dgl_graph = dgl.graph([])

    return dgl_graph


def compute_pca(data_list, esm_model, esm_tokenizer, mol2vec_model,
                n_components_esm=128, n_components_mol2vec=128):
    mol2vec_features = []
    esm_features = []

    for data in tqdm(data_list, desc='Computing PCA'):
        try:
            fields = data.strip().split("\t")
            if len(fields) != 3:
                continue
            smiles, sequences, _ = fields

            chem_feature = generate_mol2vec_feature(smiles)
            prot_feature = generate_esm_protein_feature(sequences, esm_model, esm_tokenizer)

            mol2vec_features.append(chem_feature)
            esm_features.append(prot_feature)

        except Exception as e:
            print(f"Error during PCA computation: {e}")
            continue

    mol2vec_features = np.array(mol2vec_features)
    esm_features = np.array(esm_features)

    scaler_mol2vec = StandardScaler()
    mol2vec_features = scaler_mol2vec.fit_transform(mol2vec_features)

    scaler_esm = StandardScaler()
    esm_features = scaler_esm.fit_transform(esm_features)

    pca_mol2vec = PCA(n_components=n_components_mol2vec)
    pca_mol2vec.fit(mol2vec_features)

    pca_esm = PCA(n_components=n_components_esm)
    pca_esm.fit(esm_features)

    return pca_esm, pca_mol2vec, scaler_esm, scaler_mol2vec


def process_dataset_with_pca(data_list, dir_input, dataset_type, esm_model, esm_tokenizer,
                             mol2vec_model, pca_mol2vec, pca_esm, scaler_mol2vec, scaler_esm):
    processed_data = []

    for data in tqdm(data_list, desc=f'Processing {dataset_type}'):
        try:
            fields = data.strip().split("\t")
            if len(fields) != 3:
                continue
            smiles, sequences, interaction = fields

            if len(sequences) > 5000:
                sequences = sequences[:5000]

            chem_feature = generate_mol2vec_feature(smiles)
            if chem_feature is None:
                continue
            chem_feature_scaled = scaler_mol2vec.transform(chem_feature.reshape(1, -1))
            chem_feature_pca = pca_mol2vec.transform(chem_feature_scaled).squeeze().astype(np.float16)

            prot_feature = generate_esm_protein_feature(sequences, esm_model, esm_tokenizer)
            prot_feature_scaled = scaler_esm.transform(prot_feature.reshape(1, -1))
            prot_feature_pca = pca_esm.transform(prot_feature_scaled).squeeze().astype(np.float16)

            compound_graph = smile_to_graph(smiles)
            protein_graph = protein_sequence_to_graph(sequences, feature=prot_feature_pca)

            label = 1 if interaction == "1" else 0
            processed_data.append((protein_graph, compound_graph, prot_feature_pca, chem_feature_pca, label))

        except Exception as e:
            print(f"Error processing data: {e}")
            continue

    os.makedirs(dir_input, exist_ok=True)
    processed_filename = os.path.join(dir_input, f'processed_data_{dataset_type}.pkl')
    with open(processed_filename, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"{dataset_type} preprocessing completed. Saved to {processed_filename}")


class DTIDataset(Dataset):
    def __init__(self, processed_file):
        with open(processed_file, 'rb') as f:
            self.data = pickle.load(f)
        self.data = [sample for sample in self.data if sample is not None]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    protein_graphs, compound_graphs, prot_features, chem_features, labels = zip(*batch)
    return (
        protein_graphs,
        compound_graphs,
        torch.tensor(prot_features, dtype=torch.float16),
        torch.tensor(chem_features, dtype=torch.float16),
        torch.tensor(labels, dtype=torch.float32)
    )


def main():
    setup_logging()

    # Setup paths based on DATASET_NAME
    dir_input = os.path.join(DATA_BASE_DIR, f'{DATASET_NAME}/data_split/')

    print(f"Processing dataset: {DATASET_NAME}")
    print(f"Data directory: {dir_input}")

    for dataset_type in ['train', 'val', 'test']:
        dataset = os.path.join(dir_input, f'{dataset_type}.csv')
        if not os.path.exists(dataset):
            print(f"Dataset file {dataset} does not exist. Skipping.")
            continue

        data = pd.read_csv(dataset)
        data_list = data.apply(
            lambda row: f"{row['SMILES']}\t{row['Protein']}\t{row['Y']}", axis=1
        ).tolist()

        if dataset_type == 'train':
            print(f"Computing PCA parameters for {dataset_type} set...")
            pca_esm, pca_mol2vec, scaler_esm, scaler_mol2vec = compute_pca(
                data_list, esm_model, esm_tokenizer, mol2vec_model,
                n_components_esm=N_COMPONENTS_ESM, n_components_mol2vec=N_COMPONENTS_MOL2VEC
            )
            with open(os.path.join(dir_input, 'pca_esm.pkl'), 'wb') as f:
                pickle.dump(pca_esm, f)
            with open(os.path.join(dir_input, 'pca_mol2vec.pkl'), 'wb') as f:
                pickle.dump(pca_mol2vec, f)
            with open(os.path.join(dir_input, 'scaler_esm.pkl'), 'wb') as f:
                pickle.dump(scaler_esm, f)
            with open(os.path.join(dir_input, 'scaler_mol2vec.pkl'), 'wb') as f:
                pickle.dump(scaler_mol2vec, f)
        else:
            with open(os.path.join(dir_input, 'pca_esm.pkl'), 'rb') as f:
                pca_esm = pickle.load(f)
            with open(os.path.join(dir_input, 'pca_mol2vec.pkl'), 'rb') as f:
                pca_mol2vec = pickle.load(f)
            with open(os.path.join(dir_input, 'scaler_esm.pkl'), 'rb') as f:
                scaler_esm = pickle.load(f)
            with open(os.path.join(dir_input, 'scaler_mol2vec.pkl'), 'rb') as f:
                scaler_mol2vec = pickle.load(f)

        process_dataset_with_pca(
            data_list,
            dir_input,
            f'{dataset_type}_{DATASET_NAME.lower()}',
            esm_model,
            esm_tokenizer,
            mol2vec_model,
            pca_mol2vec,
            pca_esm,
            scaler_mol2vec,
            scaler_esm
        )

    print("All dataset preprocessing completed!")


if __name__ == "__main__":
    main()
