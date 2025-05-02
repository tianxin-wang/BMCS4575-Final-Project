# 🧬 BMCSE4575-Final-Project

**Spatial Transcriptomic Enhancement via Single-cell-Informed Foundation Model**

This repository contains the code and pipelines used in our final project for BMCSE4575. We investigate spatial gene expression imputation by integrating high-resolution Xenium spatial data with paired scFFPE-seq data, using both the Tangram algorithm and the scGPT foundation model.

---

## 📄 Abstract

Accurate reconstruction of gene expression across tissue architecture is a core problem in spatial biology. In this project, we aim to impute full-transcriptome spatial maps using a subset of genes measured in the Xenium platform, guided by paired single-cell FFPE RNA-seq data. We leverage Tangram, a probabilistic cell-to-space alignment method, and examine whether enriching the input space using scGPT embeddings improves mapping accuracy. Using cosine similarity and Gromov-Wasserstein distance, we benchmark the quality of gene expression recovery and spatial coherence. Results indicate that while scGPT embeddings preserve biological structure, they provide minimal improvements in gene-level imputation tasks, highlighting the robustness of Tangram’s original formulation.

---

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `preprocess.py` | Preprocesses the FFPE, Xenium, and Visium datasets. Includes normalization, gene filtering, and dimensionality reduction. |
| `scgpt_tangram_combined.py` | Main pipeline for running Tangram on scGPT embeddings, including leave-one-out gene benchmarking and spatial visualization. |
| `test_tangram.py` | Runs standard Tangram mapping directly on gene expression matrices. Includes baseline benchmarking code. |
| `test_visium_benchmark.py` | Evaluates imputed Xenium gene maps against Visium using Gromov-Wasserstein distance for spatial alignment comparison. |

---

## 📦 Dependencies

Install with:

```bash
pip install -r requirements.txt
