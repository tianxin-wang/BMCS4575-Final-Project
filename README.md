# 🧬 BMCSE4575-Final-Project

**Spatial Transcriptomic Enhancement via Single-cell-Informed Foundation Model**

This repository contains the code and pipelines used in our final project for BMCSE4575. We investigate spatial gene expression imputation by integrating high-resolution Xenium spatial data with paired scFFPE-seq data, using both the Tangram algorithm and the scGPT foundation model.

---

## Abstract

Accurately mapping gene expression at single-cell resolution across tissue is vital for understanding spatial biology. However, current spatial transcriptomics technologies trade off between gene coverage and resolution. We address this challenge by imputing transcriptome-wide spatial gene expression in Xenium datasets using paired scFFPE-seq data. Our approach leverages Tangram, a probabilistic alignment framework, and evaluates whether integrating latent representations learned by the single-cell foundation model scGPT improves imputation accuracy. Benchmarks using cosine similarity and Gromov-Wasserstein (GW) distance show that while scGPT embeddings carry useful biological information, they yield minimal improvement over standard gene expression-based Tangram mapping. The results underscore Tangram’s robustness and highlight potential limitations of large foundation models in zero-shot imputation tasks. 

---

## File Descriptions

| File | Description |
|------|-------------|
| `preprocess.py` | Preprocesses the FFPE, Xenium, and Visium datasets. Includes normalization, gene filtering, and dimensionality reduction. |
| `test_tangram.py` | Runs standard Tangram mapping directly on gene expression matrices. Includes baseline benchmarking code. |
| `scgpt_tangram_combined.py` | Main pipeline for running Tangram on scGPT embeddings, including benchmarking code. |
| `test_visium_benchmark.py` | Evaluates imputed Xenium gene maps against Visium using Gromov-Wasserstein distance for spatial alignment comparison. |

---

## Dependencies

Install in a new conda environment with:

```bash
conda create --name <env_name> --file requirements.txt
```
---

## Authors
- Qingyuan Cai
- Tianxin Wang
