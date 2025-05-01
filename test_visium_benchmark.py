# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import squidpy as sq
import seaborn as sns
import logging
from preprocess import *
from scipy import sparse, spatial

# %%
# Set loggin
logging.basicConfig(
    # filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# %%
# Load Visium data and preprocess
DATA_DIR = "./processed_data"
OUTPUT_DIR = "./visium-results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# visium_adata = load_visium_data(load_cell_type = True)
# visium_adata = quality_control_metrics(visium_adata, "visium")
# visium_adata = filter_and_normalize(visium_adata, 500, 5, 15)
# visium_adata = dimension_reduction(visium_adata, "visium")
# visium_adata.write_h5ad(os.path.join(OUTPUT_DIR, "visium_adata.h5ad"))

# %%
# Load data
DATA_DIR = "./processed_data"
visium_adata = sc.read_h5ad(os.path.join(DATA_DIR, "visium_adata.h5ad"))
visium_adata.obsm['spatial'] = visium_adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()

sq.pl.spatial_scatter(visium_adata, color = 'STXBP1', shape=None)

# %%
# Load imputed Xenium data
DATA_DIR = "./scgpt-results-hvg-concat"
xenium_projected = sc.read_h5ad(os.path.join(DATA_DIR, "scgpt_xenium_projected.h5ad"))
xenium_projected.obsm['spatial'] = xenium_projected.obs[['x_centroid', 'y_centroid']].to_numpy()

sq.pl.spatial_scatter(xenium_projected, color = 'STXBP1', shape=None)

# %%
# Perform OT mapping
import ot

# Get raw data
xenium_raw = xenium_projected.raw.to_adata()
visium_raw = visium_adata.raw.to_adata()
xenium_raw.var_names = [g.lower() for g in xenium_raw.var_names]
visium_raw.var_names = [g.lower() for g in visium_raw.var_names]

# Get spatial coordinates and distance matrices
xenium_coords = xenium_projected.obsm['spatial']
visium_coords = visium_adata.obsm['spatial']
xenium_dist = spatial.distance.cdist(xenium_coords, xenium_coords)
visium_dist = spatial.distance.cdist(visium_coords, visium_coords)
xenium_dist /= xenium_dist.max()
visium_dist /= visium_dist.max()

# %%
OUTPUT_DIR = "./visium-results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get a list of highly variable genes from Visium data
hvg = visium_adata.var[visium_adata.var.highly_variable].index.tolist()

# Sample 10 genes from highly variable genes
# np.random.seed(42)
# sampled_hvg = np.random.choice(hvg, size=10, replace=False)
sampled_hvg = ['CD4', 'STXBP1']
gw_distances = {}
sparsity_list = {}
for gene in sampled_hvg:
    
    logging.info(f"Running OT mapping on gene: {gene}")
    gene = gene.lower()
    xenium_gene = xenium_raw[:, gene].X.toarray().flatten()
    visium_gene = visium_raw[:, gene].X.toarray().flatten()
    xenium_gene = xenium_gene / np.sum(xenium_gene)
    visium_gene = visium_gene / np.sum(visium_gene)

    gw_dist = ot.gromov.gromov_wasserstein(
        xenium_dist, visium_dist,  # Source and target cost matrices
        xenium_gene, visium_gene,  # Source and target distributions
        'square_loss',  # Loss function
        log=True,  # Return optimization details
        #numItermax=1000  # Maximum number of iterations
    )
    gw_distances[gene] = gw_dist[1]['gw_dist']
    sparsity_list[gene] = 1 - np.sum(visium_gene > 0) / visium_gene.shape[0]

df = pd.DataFrame({
    'gene': list(gw_distances.keys()),
    'gw_dist': [gw_distances[g] for g in gw_distances.keys()], 
    'sparsity': [sparsity_list[g] for g in sparsity_list.keys()]
})
df.to_csv(os.path.join(OUTPUT_DIR, "gw_distances.csv"), index=False)

# %%
# Plot the relationship between GW distance and standard deviation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sparsity', y='gw_dist')
plt.xlabel('Sparsity of Visium Gene Expression')
plt.ylabel('Gromov-Wasserstein Distance')
plt.savefig(os.path.join(OUTPUT_DIR, "gw_distances.png"), dpi=300)
plt.close()
