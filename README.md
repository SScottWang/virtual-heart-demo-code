# 4D Virtual Heart: Proof-of-Concept Demo

This repository contains the source code and analysis pipeline for the **"Towards a 4D Virtual Heart"** proposal.

🔗 **Interactive Demo Website**: [virtual-heart-demo](https://SScottWang.github.io/virtual-heart-demo/)

## 📂 Project Structure
* `notebooks/`: Jupyter notebooks demonstrating the step-by-step workflow.
    * `01_tokenizing_data_optimized.ipynb`: Geneformer tokenization and homology mapping.
    * `02_extract_embeddings_optimized.ipynb`: Embedding space visualization and bias correction (regressing out n_counts).
    * `03_in_silico_perturbation_optimized.ipynb`: In silico perturbation of Nkx2-5/Bmpr2 and Impact Score calculation.

## 🛠️ Key Technologies
* **Model**: Geneformer (Hugging Face)
* **Framework**: PyTorch, Scanpy
* **Visualization**: Plotly (3D), Matplotlib

## 🚀 Reproduction
To reproduce the environment:
```bash
pip install -r requirements.txt