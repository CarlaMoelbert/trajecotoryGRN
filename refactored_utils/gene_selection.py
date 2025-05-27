import numpy as np
import pandas as pd
from .utils import read_file
import scipy.sparse as sp
import scanpy as sc

def get_expression(file, meta=None, annotation_column='annotation', celltypes=['B1', 'B2', 'Int'], filtering=True, min_cells=10, min_genes=200):
    adata = read_file(file)
    if filtering:
        adata = filter_cells(adata, min_genes, min_cells)
    if annotation_column in adata.obs.columns:
        adata = adata[adata.obs[annotation_column].isin(celltypes)].copy()
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    if meta is not None:
        add_columns_to_exp(adata, meta)
    adata.var['gene'] = adata.var.index.copy()
    return adata

def filter_cells(adata, min_genes, min_cells, filter_mito=False, mito_threshold=10):
    """Filter cells based on number of genes, mitochondrial content, and gene counts."""
    if filter_mito:
        adata.var['mt'] = adata.var_names.str.startswith('mt-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        adata = adata[adata.obs['pct_counts_mt'] < mito_threshold].copy()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_counts=min_cells)
    return adata.copy()


def add_mean_expression(df, nodes, cells, cellstates, name='Expression'):
    """Calculate mean and variance of expression for nodes."""
    df = df[nodes.loc[nodes.gene.isin(df.columns), 'gene']]
    for label in cellstates:
        cell_set = cells.loc[cells.annotation == label, 'cell']
        X = df.loc[cell_set]
        nodes[f'{name}_Mean_{label}'] = X.mean(axis=0).values
        nodes[f'{name}_Var_{label}'] = np.var(X, axis=0).values
    return nodes.fillna(0)