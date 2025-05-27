import os
import numpy as np
import pandas as pd
import scanpy as sc
from .motif_enrichment import get_dict_motif2TF

def read_file(file):
    """Read an h5ad file and create a raw_count layer if not present.
    
    Parameters:
    - file: str, path to the h5ad file
    
    Returns:
    - Anndata object with 'raw_count' layer containing the initial data in .X
    """
    try:
        adata = sc.read_h5ad(file)
        if 'raw_count' not in adata.layers:
            adata.layers['raw_count'] = adata.X.copy()
    except OSError as e:
        print(f'Error reading the file {file}: {e}')
        raise
    return adata

def intersect(list1, list2):
    """Return the intersection of two lists as a list of unique elements."""
    return list(set(list1) & set(list2))

def add_TF_information(annotation, motifs):
    """Add transcription factor (TF) information to the annotation data."""
    dic_motif2TFs = get_dict_motif2TF(motifs)
    tfs = set.union(*map(set, dic_motif2TFs.values())).intersection(annotation.gene)
    annotation['TF'] = annotation.gene.isin(tfs)
    return annotation

def combine_all(sub_grns):
    combined_df = None
    for cell_state, edges_df in sub_grns.items():
        edges_df = edges_df.rename(columns={'Coef': f'Coef_{cell_state}', 'Score': f'Score_{cell_state}', 'Weight': f'Weight_{cell_state}', 'Coexpr': f'Coexpr_{cell_state}'})
        combined_df = edges_df if combined_df is None else combined_df.merge(edges_df, on=['Source', 'Target'], how='outer')
    coef_columns = [col for col in combined_df.columns if 'Coef' in col]
    combined_df['Coef'] = combined_df[coef_columns].max(axis=1)
    weight_columns = [col for col in combined_df.columns if 'Weight' in col]
    combined_df['Weight'] = combined_df[coef_columns].max(axis=1)
    return combined_df

def select_top_n(df, column="Coef", n=1000):
    # Filter the column to include only values greater than 0
    filtered_df = df[df[column] > 0]
    top_n = filtered_df.nlargest(n, column)

    filtered_df = df[df[column] < 0]
    bottom_n = filtered_df.nsmallest(n, column)

    combined_df = pd.concat([top_n, bottom_n])
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df

def get_expr_matrix(expr, todense=True, layer='normalized_count_1'):
    """Return expression matrix in dense format if requested."""
    e = expr.layers[layer]
    if todense:
        e = e.toarray()
    return pd.DataFrame(e, index=expr.obs.index, columns=expr.var.index)
