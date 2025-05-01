'''
This script performs basic preprocessing steps on the Xenium and FFPE data.
'''

# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from scipy import sparse
import seaborn as sns
import logging

# %%
# Set paths
DATA_DIR = "./outs"
OUTPUT_DIR = "./processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

logging.basicConfig(
    # filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_xenium_data(load_cell_type = True):
    """Load the Xenium cell feature matrix from h5 file."""
    
    logging.info("Loading Xenium cell feature matrix...")
    adata = sc.read_10x_h5(os.path.join(DATA_DIR, "cell_feature_matrix.h5"))
    cell_info = pd.read_csv(os.path.join(DATA_DIR, "cells.csv.gz"))
    cell_info.index = cell_info['cell_id']
    cell_info.index.name = None
    adata.obs = cell_info
    
    if load_cell_type:
        logging.info("Loading cell type annotations...")
        cell_types_path = os.path.join(DATA_DIR, "Cell_Barcode_Type_Matrices.xlsx")
        cell_types = pd.read_excel(cell_types_path, sheet_name="Xenium R1 Fig1-5 (supervised)")
        cell_types.columns = ["cell_id", "cell_type"]
        adata.obs = adata.obs.merge(cell_types, on="cell_id", how="left")
    
    adata.obs_names = adata.obs["cell_id"].astype(str).values
    adata.obs_names.name = None

    logging.info(f"Loaded xenium data with shape: {adata.shape}")
    return adata

def load_visium_data(load_cell_type = True):
    """Load the Visium cell feature matrix from h5 file."""
    
    logging.info("Loading Visium cell feature matrix...")
    adata = sc.read_10x_h5(os.path.join(DATA_DIR, "CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"))
    adata.var_names_make_unique()
    cell_info = pd.read_csv(os.path.join(DATA_DIR, "tissue_positions.csv"))
    cell_info.index = cell_info['barcode'].values
    cell_info.index.name = None
    adata.obs = adata.obs.join(cell_info)
    
    if load_cell_type:
        logging.info("Loading cell type annotations...")
        cell_types_path = os.path.join(DATA_DIR, "Cell_Barcode_Type_Matrices.xlsx")
        cell_types = pd.read_excel(cell_types_path, sheet_name="Visium")
        cell_types.columns = ["barcode", "cluster", "cell_type"]
        adata.obs = adata.obs.merge(cell_types, on="barcode", how="left")

    logging.info(f"Loaded visium data with shape: {adata.shape}")
    return adata

def load_ffpe_data(load_cell_type = True):    
    """Load the scFFPE-seq cell feature matrix from h5 file."""

    logging.info("Loading FFPE cell feature matrix...")
    adata = sc.read_10x_mtx(os.path.join(DATA_DIR, "sample_filtered_feature_bc_matrix"))
    adata.obs["barcode"] = adata.obs.index
    
    if load_cell_type:
        logging.info("Loading cell type annotations...")
        cell_types_path = os.path.join(DATA_DIR, "Cell_Barcode_Type_Matrices.xlsx")
        cell_types = pd.read_excel(cell_types_path, sheet_name="scFFPE-Seq")
        cell_types.columns = ["barcode", "cell_type"]
        adata.obs = adata.obs.merge(cell_types, on="barcode", how="left")
    
    adata.obs_names = adata.obs["barcode"].values
    
    logging.info(f"Loaded FFPE data with shape: {adata.shape}")
    return adata

# %%
def quality_control_metrics(adata, tech = "xenium", output_dir = OUTPUT_DIR):
    """Calculate quality metrics on the AnnData object."""
    logging.info("Calculating quality metrics...")
    
    # Calculate quality metrics including mitochondrial genes
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    """Plot quality control metrics."""
    logging.info("Plotting QC metrics...")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    
    # Plot 1: Genes per cell
    sns.histplot(adata.obs['n_genes_by_counts'], kde=False, ax=axes[0])
    axes[0].set_title('Genes per cell')
    axes[0].set_xlabel('Number of genes')
    axes[0].set_ylabel('Count')
    
    # Plot 2: Counts per cell
    sns.histplot(adata.obs['total_counts'], kde=False, ax=axes[1])
    axes[1].set_title('Counts per cell')
    axes[1].set_xlabel('Total counts')
    axes[1].set_ylabel('Count')
    
    # Plot 3: Mitochondrial counts per cell
    sns.histplot(adata.obs['pct_counts_mt'], kde=False, ax=axes[2])
    axes[2].set_title('Mitochondrial counts per cell')
    axes[2].set_xlabel('Mitochondrial counts')
    axes[2].set_ylabel('Count')
    
    plt.savefig(os.path.join(output_dir, f'quality_control_metrics_{tech}.png'), dpi=300, bbox_inches='tight')
    logging.info(f"Quality control metrics plot saved to {os.path.join(output_dir, f'quality_control_metrics_{tech}.png')}")
    
    return adata

# %%
def filter_and_normalize(adata, min_genes=10, min_cells=5, max_mt_percent=-1):
    """
    Filter cells based on quality metrics.
    Args:
        adata (AnnData): The AnnData object to filter and normalize.
        min_genes (int): The minimum number of genes a cell must have.
        min_cells (int): The minimum number of cells a gene must have.
        max_mt_percent (int): The maximum percentage of mitochondrial genes a cell can have. (default: -1, no filtering)
    Returns:
        adata (AnnData): The filtered and normalized AnnData object.
    """
    logging.info(f"Filtering cells with at least {min_genes} genes, genes expressed in at least {min_cells} cells...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Filter cells based on mitochondrial percentage
    if max_mt_percent > 0:
        adata = adata[adata.obs['pct_counts_mt'] < max_mt_percent].copy()
    logging.info(f"Filtered data with shape: {adata.shape}")
    
    # Save original count matrix in a layer
    logging.info("Saving original count matrix in a layer...")
    adata.layers['counts'] = adata.X.copy()
    
    # Normalize data
    logging.info("Normalizing data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Log transform
    logging.info("Log transforming data...")
    sc.pp.log1p(adata)
    
    return adata

# %%
def dimension_reduction(adata, tech = "xenium", output_dir = OUTPUT_DIR):
    """Perform highly variable gene selection and dimension reduction on the AnnData object."""
    
    # Find highly variable genes
    logging.info("Finding highly variable genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    
    # Keep only highly variable genes
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    
    # Regress out total_counts
    logging.info("Regressing out total_counts...")
    sc.pp.regress_out(adata, ['total_counts'])
    
    # Scale data
    logging.info("Scaling data...")
    sc.pp.scale(adata, max_value=10)
    
    # Run PCA
    logging.info("Running PCA...")
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca(adata, show=False, title='PCA')

    # Compute neighborhood graph
    logging.info("Computing neighborhood graph...")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
    
    # Run UMAP
    logging.info("Running UMAP...")
    sc.tl.umap(adata)
    sc.pl.umap(adata, show=False, title='UMAP')
    plt.savefig(os.path.join(output_dir, f'umap_{tech}.png'), dpi=300, bbox_inches='tight')
    sc.pl.umap(adata, show=False, color='cell_type', title='UMAP')
    plt.savefig(os.path.join(OUTPUT_DIR, f'umap_{tech}_cell_type.png'), dpi=300, bbox_inches='tight')
  
    return adata