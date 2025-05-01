"""
This script performs Tangram mapping between preprocessed Xenium and single-cell data.
It also evaluates the mapping accuracy on leave-out genes.
"""

# %%
import os
import torch
# import faiss
import warnings
import logging
import scanpy as sc
import tangram as tg
import squidpy as sq
import numpy as np
import scgpt as scg
import matplotlib.pyplot as plt
from preprocess import *
from pathlib import Path

warnings.filterwarnings('ignore')

# %%
# Set logging
logging.basicConfig(
    # filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "./processed_data"
OUTPUT_DIR = "./tangram-results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
def run_tangram(adata_ref, adata_spatial, leave_out_gene=None):
    """
    Run Tangram to map single-cell reference data to spatial data.
    """
    logging.info("Running Tangram mapping...")
     
    if leave_out_gene is not None:
        adata_spatial = adata_spatial[:, adata_spatial.var_names!=leave_out_gene]
    
    # Prepare the data for Tangram
    logging.info("Preparing data for Tangram...")
    tg.pp_adatas(adata_ref, adata_spatial, genes=None)
        
    # Run Tangram
    logging.info("Mapping cells to space...")
    adata_map = tg.map_cells_to_space(
        adata_sc=adata_ref, 
        adata_sp=adata_spatial,
        mode='cells',
        density_prior='uniform',
        num_epochs=1200,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    adata_map.obs.index.name = None
    adata_map.var.index.name = None

    # Project the reference data to the spatial data
    logging.info("Projecting genes...")
    adata_ge = tg.project_genes(adata_map, adata_ref)
    adata_ge.obs.index.name = None
    adata_ge.var.index.name = None

    # Plot the mapping score
    plt.figure(figsize=(10, 8))
    tg.plot_training_scores(adata_map)
    if leave_out_gene is not None:
        plt.savefig(os.path.join(OUTPUT_DIR, f"tangram_training_scores_{leave_out_gene}.png"), dpi=300)
    else:
        plt.savefig(os.path.join(OUTPUT_DIR, "tangram_training_scores.png"), dpi=300)
    plt.close()
    
    logging.info("Tangram mapping completed successfully!")
    return adata_ge, adata_map

# %%
def evaluate_tangram_mapping(ground_truth_adata, mapped_adata, test_gene):
    """
    Evaluate Tangram mapping accuracy using cosine similarity  on mapped genes
    """
    # Get original expression values for mapped genes
    ground_truth_adata.var.index = [g.lower() for g in ground_truth_adata.var.index]
    mapped_adata.var.index = [g.lower() for g in mapped_adata.var.index]
    test_gene = test_gene.lower()
    gt_expr = ground_truth_adata[:, test_gene].X.toarray()
    imp_expr = mapped_adata[:, test_gene].X.toarray()
    test_cosine_sim = np.dot(gt_expr.flatten(), imp_expr.flatten()) / (np.linalg.norm(gt_expr.flatten()) * np.linalg.norm(imp_expr.flatten()))
    sp_sparsity = 1 - np.sum(gt_expr > 0) / gt_expr.shape[0]

    plt.figure(figsize=(6, 6))
    plt.scatter(gt_expr, imp_expr, alpha=0.5, s=1)
    max_val = max(gt_expr.max(), imp_expr.max())  # Add diagonal line
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    plt.text(0.1, 0.9, f'cos similarity = {test_cosine_sim:.3f} \n sparsity = {sp_sparsity:.3f}', 
            transform=plt.gca().transAxes, fontsize=12)  # Add correlation annotation
    plt.xlabel('Ground Truth Expression')
    plt.ylabel('Imputed Expression')
    plt.title(test_gene)
    plt.savefig(os.path.join(OUTPUT_DIR, f"gene_correlation_{test_gene}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate median correlation on the remaining genes
    ground_truth_genes = list(set(ground_truth_adata.var_names).intersection(mapped_adata.var_names))
    ground_truth_genes.remove(test_gene)
    ground_truth_expr = ground_truth_adata[:, ground_truth_genes].X.toarray()
    imputed_expr = mapped_adata[:, ground_truth_genes].X.toarray()
    gene_cos_sim = {}
    gene_sparsity = {}
    for i, gene in enumerate(ground_truth_genes):
        gt_expr = ground_truth_expr[:, i]
        imp_expr = imputed_expr[:, i]
        cosine_sim = np.dot(gt_expr.flatten(), imp_expr.flatten()) / (np.linalg.norm(gt_expr.flatten()) * np.linalg.norm(imp_expr.flatten()))
        gene_cos_sim[gene] = cosine_sim
        gene_sparsity[gene] = 1 - np.sum(gt_expr > 0) / gt_expr.shape[0]

    median_cos_sim = np.median(list(gene_cos_sim.values()))
    print(f"Median cosine similarity across genes: {median_cos_sim:.3f}")
    
    plt.figure(figsize=(8, 6))
    plt.violinplot(list(gene_cos_sim.values()))
    plt.ylabel('Cosine Similarity')
    plt.title('Distribution of Gene-wise Cosine Similarity')
    plt.axhline(y=median_cos_sim, color='r', linestyle='--', label=f'Median={median_cos_sim:.3f}')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"ref_gene_cos_sim_{test_gene}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(list(gene_sparsity.values()), list(gene_cos_sim.values()))
    plt.xlabel('Sparsity')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity vs Sparsity')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"ref_gene_sparsity_cos_sim_{test_gene}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    return test_cosine_sim, gene_cos_sim

if __name__ == "__main__":
    # # %%
    # # Load data
    # xenium_adata = load_xenium_data(load_cell_type = True)
    # xenium_adata = quality_control_metrics(xenium_adata, "xenium")

    # ffpe_adata = load_ffpe_data(load_cell_type = True)
    # ffpe_adata = quality_control_metrics(ffpe_adata, "ffpe")

    # # %%
    # # Filter and normalize data
    # xenium_adata = filter_and_normalize(xenium_adata, 10, 5)
    # ffpe_adata = filter_and_normalize(ffpe_adata, 500, 5, 15)

    # # %%
    # # Visualize data
    # xenium_adata = dimension_reduction(xenium_adata, "xenium")
    # ffpe_adata = dimension_reduction(ffpe_adata, "ffpe")

    # # %%
    # # Save data
    # xenium_adata.write_h5ad(os.path.join(DATA_DIR, "xenium_adata.h5ad"))
    # ffpe_adata.write_h5ad(os.path.join(DATA_DIR, "ffpe_adata.h5ad"))

    # # %%
    # # Sample 20000 cells from xenium data
    # xenium_adata = sc.read_h5ad(os.path.join(DATA_DIR, "xenium_adata.h5ad"))
    # xenium_adata = xenium_adata.raw.to_adata()
    # sample = np.random.choice(range(xenium_adata.shape[0]), size=20000, replace=False)
    # xenium_adata_subsampled = xenium_adata[sample, :]
    # xenium_adata_subsampled = dimension_reduction(xenium_adata_subsampled, "xenium_subsampled")
    # xenium_adata_subsampled.write_h5ad(os.path.join(DATA_DIR, "xenium_adata_subsampled.h5ad"))

    # # %%
    # # Plot the subsampled xenium data
    # ## Generate the coordinates
    # xs = xenium_adata_subsampled.obs.x_centroid.values
    # ys = xenium_adata_subsampled.obs.y_centroid.values

    # ## Create a categorical color map for cell types
    # unique_cell_types = xenium_adata_subsampled.obs.cell_type.unique()
    # cell_type_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cell_types)))
    # cell_type_cmap = dict(zip(unique_cell_types, cell_type_colors))

    # ## Plot the subsampled xenium data with cell types
    # plt.scatter(xs, ys, s=.5, c=[cell_type_cmap[ct] for ct in xenium_adata_subsampled.obs.cell_type])
    # plt.legend(handles=[plt.scatter([], [], c=color, label=ct) for ct, color in cell_type_cmap.items()], bbox_to_anchor=(1.5, 1), loc='upper right')
    # plt.show()
    # plt.savefig(os.path.join(DATA_DIR, "xenium_subsampled_spatial_data_cell_type.png"), dpi=300)

    # %%
    # Run Tangram mapping on all genes
    xenium_adata_subsampled = sc.read_h5ad(os.path.join(DATA_DIR, "xenium_adata_subsampled.h5ad"))
    ffpe_adata = sc.read_h5ad(os.path.join(DATA_DIR, "ffpe_adata.h5ad"))

    adata_ge, adata_map = run_tangram(
        ffpe_adata.raw.to_adata(), 
        xenium_adata_subsampled.raw.to_adata()
    )

    adata_map.write_h5ad(os.path.join(OUTPUT_DIR, "tangram_map.h5ad"))
    adata_ge = dimension_reduction(adata_ge, "tangram", OUTPUT_DIR)
    adata_ge.write_h5ad(os.path.join(OUTPUT_DIR, "xenium_projected.h5ad"))

    # %%
    # Run Tangram mapping on a leave-one-out basis
    xenium_adata_subsampled = sc.read_h5ad(os.path.join(DATA_DIR, "xenium_adata_subsampled.h5ad"))
    ffpe_adata = sc.read_h5ad(os.path.join(DATA_DIR, "ffpe_adata.h5ad"))
    
    sample_size = 10
    genes = list(xenium_adata_subsampled.var_names.intersection(ffpe_adata.var_names))
    sampled_genes = np.random.choice(genes, size=sample_size, replace=False)

    # %%
    # Run Tangram mapping on a leave-one-out basis
    for gene in sampled_genes:
        # Get genes excluding current test gene
        
        logging.info(f"Running Tangram mapping on leave-one-out gene: {gene}")
        # Run mapping with subset of genes
        adata_ge_loo, adata_map_loo = run_tangram(
            adata_ref = ffpe_adata.raw.to_adata(),
            adata_spatial = xenium_adata_subsampled.raw.to_adata(),
            leave_out_gene = gene
        )
        
        test_cosine_sim, gene_cos_sim = evaluate_tangram_mapping(
            ground_truth_adata = xenium_adata_subsampled.raw.to_adata(),
            mapped_adata = adata_ge_loo,
            test_gene = gene
        )