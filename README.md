# MMTF-DTI

Multi-Modal Transformer Fusion for Drug-Target Interaction Prediction

## Requirements

- Python 3.9
- PyTorch 2.0+ (CUDA 11.8)
- DGL 1.0+
- Transformers 4.40+
- RDKit
- Gensim 3.8+
- Scikit-learn
- DGLLife
- Mol2Vec

### Preprocessing

1. Download pretrained models:
   - ESM-2: [esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
   - Mol2Vec: [model_300dim.pkl](https://github.com/samoturk/mol2vec)

2. Configure dataset in `datapre.py`:

   ```python
   DATASET_NAME = 'DrugBank'  # Options: 'DrugBank', 'BindingDB', 'Human', 'Celegans'
   DATA_BASE_DIR = './data/'
   MOL2VEC_PATH = 'model_300dim.pkl'
   ESM_PATH = 'esm2_t12_35M_UR50D'

3. Run preprocessing:

```python
datapre.py
```

4. Generated files:

   - processed_data_train_*.pkl

   - processed_data_val_*.pkl

   - processed_data_test_*.pkl

   - PCA and scaler models

### Single-Domain Training

Train and evaluate on a single dataset:

```python
main.py
```

### Cross-Domain Training

```python
Cross_domain.py
```

