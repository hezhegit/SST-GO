<h1 align="center">
SGAFGO: Fusion of Multi-Scale Sequence and Multi-Track Graph Networks for Protein Function Prediction
</h1>

<p align="center">
<img src="https://img.shields.io/badge/OS-Ubuntu22.04-blue" />
<img src="https://img.shields.io/badge/Python-%3E%3D3.8-red" />
<img src="https://img.shields.io/badge/Build-Success-green" />
<img src="https://img.shields.io/badge/Release-0.1-blue" />
</p>
<p align="center">
This repository contains scripts used to build and train the SGAFGO model, together with scripts for evaluating model performance.
</p>

---

## Datasets

This repository uses benchmark datasets for training and evaluating SGAFGO.

Dataset snapshots and train/validation/test splits follow the protocols adopted in previous works.

**This work adopts the datasets and split strategies commonly used in DeepFRI, HEAL and TAWFN.**

Representative references:

- **DeepFRI**: Gligorijević, V.; Renfrew, P. D.; Kosciolek, T.; et al. *Structure-based protein function prediction using graph convolutional networks.* **Nat. Commun.** **2021**, *12*, 3168.
- **HEAL**: Ning, Q.; Yang, R.; Sun, Z.; et al. *HEAL: Hierarchical Enhanced Attention Learning for Protein Function Prediction.* **Bioinformatics** **2023**, *39*, i17-i24.
- **TAWFN**: Meng, L.; Wang, X.; et al. *TAWFN: a deep learning framework for protein function prediction.* **Bioinformatics** **2024**, *40*, btae571.


All dataset files should be placed under the `.data/` directory.

---

## Environment Setup

### Create conda environment

```bash
conda create -n sgafgo python=3.8 -y
conda activate sgafgo
```


Install Python dependencies:

```bash
pip install numpy pandas scikit-learn biopython tqdm
pip install transformers==4.5.1
pip install torch==1.8.0 torchvision==0.9.0
```

**Note:** Choose a PyTorch wheel matching your CUDA version if using GPU.


------

## Usage

### Step 1: Feature extraction

Download pretrained ProtT5 models, then extract sequence embeddings using the scripts in `feature_extract`:

```bash
python feature_extract/extract_pt5.py.py
```


### Step 2: Training and prediction

Configure `config.py` with data, feature, and output paths and hyperparameters.

Train:

```bash
python train.py
```

Evaluate / Predict:

```bash
python test.py
```
