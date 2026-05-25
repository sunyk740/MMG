# MMG
**MMG (Meta Marker Generator)** is an interpretable linear framework for cross-modal and cross-species single-cell data integration.

## 📖 Overview

MMG learns concise, linearly weighted gene combinations called **"meta-markers"** through a denoising autoencoder with adversarial domain adaptation. Unlike black-box deep learning approaches, MMG produces interpretable gene weight vectors that capture essential cellular identity information while remaining robust to technical artifacts.

## 📦 Installation

```bash
pip install mmg-sc
```

## 🚀 Quick Start

```bash
import mmg_sc as mmg
```

## 📓 Tutorials

Detailed step-by-step tutorials are available in the examples/ directory:

examples/cross_species_integration.ipynb 
examples/cross_species_transfer.ipynb 
