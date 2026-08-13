<div align="center">
  
  # HyperID-NeuroSymbolic

  **Ontology-Grounded Explainable Breed-Ancestry Estimation for Mixed and Indigenous Companion Animals**

  [![Paper](https://img.shields.io/badge/IEEE-Paper_Coming_Soon-blue.svg)](https://ieeexplore.ieee.org/)
  [![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

---

## 📖 Overview

**HyperID** is an open-source, neuro-symbolic AI pipeline designed to accurately and transparently classify mixed-breed and indigenous companion animals. Traditional Convolutional Neural Networks (CNNs) often act as "black boxes" and struggle with out-of-distribution strays or highly hybridized breeds. 

HyperID solves this by merging the representation power of Deep Learning with the logic of Knowledge Engineering. It employs a **Multi-Task Trait Network** to extract physical properties (Ear Shape, Coat Type, Size, etc.) and fuses it with a **Zero-Shot CLIP Vision-Language Model** in log-space. Finally, an **OWL Ontology (SPARQL)** reasons over these extracted traits to predict the breed ancestry with complete, human-readable explainability.

---

## ✨ Key Features

- 🧠 **Multi-Task Neural Network:** Simultaneously predicts 6 distinct physical traits instead of a single opaque breed class.
- 🔗 **Zero-Shot Textual Fusion:** Leverages OpenAI's CLIP to inject zero-shot vision-language knowledge into the visual pipeline, correcting CNN dataset biases.
- 🦉 **OWL Ontology Grounding:** Replaces the standard neural classification head with a symbolic SPARQL reasoner. The model doesn't just guess; it checks a rulebook.
- 🔍 **100% Explainability:** Outputs the exact physical traits that led to the breed decision, providing transparent "trait traces" for veterinary and shelter use.
- 🐕 **Out-of-Distribution Generalization:** Capable of elegantly handling new mixed breeds (e.g., Gerberian Shepsky) without requiring total model retraining.

---

## 📊 Dataset & Resources

- **Dataset:** This project utilizes a specialized subset of the [Oxford-IIIT Pet Dataset (Kaggle)](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset) alongside custom collected mixed-breed imagery.
- **Ontology:** The official Breed Trait Ontology (TTL format) is included in the repository.
- **Publication:** *Placeholder for IEEE Conference Link*

---

## 📁 Repository Structure

This repository has been streamlined to contain only the core components of the HyperID pipeline:

- **`train_trait_model_fact_checked_v02.py`**: The primary training script for the Multi-Task Trait Network. It reads the ontology, builds the 6-head neural network, and trains it using early stopping and PyTorch optimizations.
- **`crosscheck_traits_ensemble.py`**: The main inference pipeline. It takes an input image, runs the visual trait extraction, fuses it with CLIP zero-shot features, and queries the ontology for the final breed prediction.
- **`hyperid_reasoning.py`**: The symbolic logic engine. It contains the SPARQL queries that interact with the OWL file to map physical traits to breed ancestry.
- **`train_cnn_baseline.py`**: The standard ResNet-18 baseline script, provided for direct performance comparison against the neuro-symbolic approach.
- **`hyperid_clip_baseline.py`**: The standalone zero-shot CLIP baseline script.
- **`breed_trait_ontology_fact_checked_v02.ttl`**: The verified Web Ontology Language (OWL) knowledge base defining the specific physical traits of the 13 training breeds.
- **`mixed-breed/`**: A curated folder containing specific test images of out-of-distribution mixed-breed dogs (like Goldadors and Gerberian Shepskys) used to evaluate zero-shot reasoning generalization.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/HyperID-NeuroSymbolic.git
   cd HyperID-NeuroSymbolic
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   Install PyTorch according to your CUDA version, then install the remaining requirements:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers rdflib scikit-learn matplotlib numpy
   ```

---

## 🚀 Usage

### 1. Training the Multi-Task Trait Network
To train the physical trait extractor from scratch using the Oxford dataset and the verified ontology:

```bash
python train_trait_model_fact_checked_v02.py \
    --ontology breed_trait_ontology_fact_checked_v02.ttl \
    --data ./data \
    --epochs 30 \
    --batch-size 64
```
*Note: The script automatically precomputes frozen CLIP embeddings to accelerate training.*

### 2. Running Inference & Ontology Reasoning
To test an image through the full neuro-symbolic pipeline (Neural extraction $\rightarrow$ CLIP Fusion $\rightarrow$ SPARQL Reasoning):

```bash
python crosscheck_traits_ensemble.py \
    --image path/to/dog.jpg \
    --ontology breed_trait_ontology_fact_checked_v02.ttl \
    --model trained_model_fact_checked/multitask_trait_model.pt
```

### 3. Generating Paper Figures
To reproduce the exact training curves, trait accuracy bars, and ablation studies used in the paper:

```bash
python generate_all_paper_plots.py
```
*(Ensure training has been completed first, as it reads from the generated `training_loss_log.json`)*

---

## 📈 Methodology Workflow

The architecture processes the image through two parallel branches (Neural Trait Net and Textual CLIP). The ensemble logits are extracted as a discrete Trait Profile (e.g., `FloppyEars`, `MediumSize`), which is then queried against the OWL Ontology using SPARQL to deduce the final breed matching score.

---

## 🤝 Contributing

Contributions to expand the breed ontology, improve the multi-task CNN trunk, or add more indigenous stray classes are highly welcome! Please submit a Pull Request or open an Issue for discussion.

## 🏆 Acknowledgements & Credits

This project was developed by the **StrayCare Team (Team Edgeventurers)** as part of our mission to advance animal welfare and veterinary technologies.

We proudly build upon the following open-source frameworks and research:

<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Meta_AI-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="Meta AI" />
  <img src="https://img.shields.io/badge/OpenAI_CLIP-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI CLIP" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
</p>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed for Animal Welfare and Veterinary applications by Team Edgeventurers.*
