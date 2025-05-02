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
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')

# %%
# Set logging
logging.basicConfig(
    # filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DATA_DIR = "./processed_data"
OUTPUT_DIR = "./scgpt-results-hvg-concat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%
def run_scGPT(adata, type="ffpe", leave_out_gene=None):
    """
    Run scGPT embedding on the given AnnData object.
    """
    logging.info("Running scGPT embedding...")
    
    model_dir = Path("./scGPT_human")
    adata.var["index"] = adata.var_names.to_list()
    cell_type_key = "cell_type"

    embed_adata = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col="index",
        obs_to_save=cell_type_key,  # optional arg, for saving metainfo
        batch_size=64,
        return_new_adata=True,
    )
    
    sc.pp.neighbors(embed_adata, use_rep="X", n_neighbors=10, n_pcs=50)
    sc.tl.umap(embed_adata)
    
    if leave_out_gene is None:
        sc.pl.umap(embed_adata, color=cell_type_key, frameon=False, wspace=0.4)
        plt.savefig(os.path.join(OUTPUT_DIR, f"umap_scgpt_embedding_{type}.png"), 
                    dpi=300, bbox_inches="tight")

    adata.obsm["X_scgpt"] = embed_adata.X
    adata.obsm["X_scgpt_umap"] = embed_adata.obsm["X_umap"]
    
    logging.info("scGPT embedding completed successfully!")
    return adata

# %%
def run_tangram(adata_ref_latent, adata_spatial_latent, adata_ref, leave_out_gene=None):
    """
    Run Tangram to map single-cell reference data to spatial data.
    """
    logging.info("Running Tangram mapping...")
    
    # Prepare the data for Tangram
    logging.info("Preparing data for Tangram...")
    tg.pp_adatas(adata_ref_latent, adata_spatial_latent, genes=None)
    
    # Run Tangram
    logging.info("Mapping cells to space...")
    adata_map = tg.map_cells_to_space(
        adata_sc=adata_ref_latent, 
        adata_sp=adata_spatial_latent,
        mode='cells',
        density_prior='uniform',
        num_epochs=1200,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    logging.info(f"Leave out gene: {leave_out_gene}, Adata_map shape: {adata_map.X.shape}")
    adata_map.obs.index.name = None
    adata_map.var.index.name = None

    # Project the reference data to the spatial data
    logging.info("Projecting genes...")
    projected_genes = adata_map.X.T @ adata_ref.X
    adata_ge = sc.AnnData(X=projected_genes, obs=adata_map.var.copy(), var=adata_ref.var.copy())
    logging.info(f"Leave out gene: {leave_out_gene}, Adata_ge shape: {adata_ge.X.shape}")
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
def run_latent_tangram(adata_ref, adata_spatial, leave_out_gene=None):
    """
    Run Tangram mapping on the latent space of scGPT embeddings.
    """
    logging.info(f"adata_ref shape: {adata_ref.X.shape}")
    if leave_out_gene is not None:
        adata_spatial = adata_spatial[:, adata_spatial.var_names!=leave_out_gene]
    
    # Run scGPT embedding for FFPE and Xenium data
    ffpe_scgpt_embedding = run_scGPT(adata_ref, "ffpe", leave_out_gene)
    xenium_scgpt_embedding = run_scGPT(adata_spatial, "xenium", leave_out_gene)

    L_ref = ffpe_scgpt_embedding.obsm["X_scgpt"]  # (cells × latent_dim)
    L_spatial = xenium_scgpt_embedding.obsm["X_scgpt"]
    latent_gene_names = [f"latent_{i}" for i in range(L_ref.shape[1])]
    
    logging.info("Concatenate the scGPT latent space with the gene expression data")
    ref_concatenated = np.concatenate([L_ref, ffpe_scgpt_embedding.X.toarray()], axis=1)
    spatial_concatenated = np.concatenate([L_spatial, xenium_scgpt_embedding.X.toarray()], axis=1)
    
    adata_ref_latent = sc.AnnData(X=ref_concatenated, obs=ffpe_scgpt_embedding.obs.copy())
    adata_spatial_latent = sc.AnnData(X=spatial_concatenated, obs=xenium_scgpt_embedding.obs.copy())
    # Rename scGPT latent space dimensions to fake "genes"
    adata_ref_latent.var_names = latent_gene_names + ffpe_scgpt_embedding.var_names.to_list()
    adata_spatial_latent.var_names = latent_gene_names + xenium_scgpt_embedding.var_names.to_list()
    
    # adata_ref_latent = sc.AnnData(X=L_ref, obs=ffpe_scgpt_embedding.obs.copy())
    # adata_spatial_latent = sc.AnnData(X=L_spatial, obs=xenium_scgpt_embedding.obs.copy())
    # adata_ref_latent.var_names = latent_gene_names
    # adata_spatial_latent.var_names = latent_gene_names
    
    # Tangram mapping
    logging.info("Mapping scRNA to Xenium latent space locations using Tangram...")
    adata_ge, adata_map = run_tangram(
        adata_ref_latent,
        adata_spatial_latent,
        adata_ref,
        leave_out_gene=leave_out_gene
    )
    
    # Store only the all gene mapping adata results
    if leave_out_gene is None:
        ffpe_scgpt_embedding.write(os.path.join(OUTPUT_DIR, "ffpe_scgpt_embedding.h5ad"))
        xenium_scgpt_embedding.write(os.path.join(OUTPUT_DIR, "xenium_scgpt_embedding.h5ad"))
        adata_ge = dimension_reduction(adata_ge, "scgpt_tangram", OUTPUT_DIR)
        adata_ge.write(os.path.join(OUTPUT_DIR, "scgpt_xenium_projected.h5ad"))
        adata_map.write(os.path.join(OUTPUT_DIR, "scgpt_tangram_map.h5ad"))
    
    return adata_ge, adata_map

# %%
def evaluate_tangram_mapping(ground_truth_adata, mapped_adata, test_gene):
    """
    Evaluate Tangram mapping accuracy using cosine similarity on mapped genes
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
    logging.info(f"Median cosine similarity across genes: {median_cos_sim:.3f}")
    
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

# %%
# Read in processed FFPE and Xenium data
ffpe_adata = sc.read_h5ad(os.path.join(DATA_DIR, "ffpe_adata.h5ad"))
xenium_adata_subsampled = sc.read_h5ad(os.path.join(DATA_DIR, "xenium_adata_subsampled.h5ad"))

# %%
# Tangram mapping on all genes
adata_ref = ffpe_adata.raw.to_adata()
adata_ref = adata_ref[:, adata_ref.var.highly_variable]

adata_spatial = xenium_adata_subsampled.raw.to_adata()
adata_spatial = adata_spatial[:, adata_spatial.var.highly_variable]

logging.info("Tangram mapping on all genes...")
adata_ge, adata_map = run_latent_tangram(
    adata_ref = adata_ref,
    adata_spatial = adata_spatial
)

# %%
# Leave-One-Out benchmarking
sample_size = 10
genes = list(xenium_adata_subsampled.var_names.intersection(adata_ref.var_names))
sampled_genes = np.random.choice(genes, size=sample_size, replace=False)

# %%
# Run Tangram mapping on a leave-one-out basis
for gene in sampled_genes:
    # Get genes excluding current test gene
    logging.info(f"Running Tangram mapping on leave-one-out gene: {gene}")
    # Run mapping with subset of genes
    adata_ge_loo, adata_map_loo = run_latent_tangram(
        adata_ref = adata_ref,
        adata_spatial = adata_spatial,
        leave_out_gene = gene
    )
    
    test_cosine_sim, gene_cos_sim = evaluate_tangram_mapping(
        ground_truth_adata = adata_spatial,
        mapped_adata = adata_ge_loo,
        test_gene = gene
    )

logging.info("Finished scGPT embedding and benchmarking.")
