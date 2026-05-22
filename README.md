**MMG (Meta Marker Generator)** is an interpretable linear framework for cross-modal and cross-species single-cell data integration.

## 📖 Overview

MMG learns concise, linearly weighted gene combinations called **"meta-markers"** through a denoising autoencoder with adversarial domain adaptation. Unlike black-box deep learning approaches, MMG produces interpretable gene weight vectors that capture essential cellular identity information while remaining robust to technical artifacts.

### Key capabilities

- **Cross-modal/species label transfer**: From scRNA-seq to scATAC-seq, spatial transcriptomics, and more
- **Cross-modal/species data integration**: Mouse, human, macaque, marmoset, turtle, lizard, chimpanzee

## 📦 Installation

```bash
pip install mmg-sc
