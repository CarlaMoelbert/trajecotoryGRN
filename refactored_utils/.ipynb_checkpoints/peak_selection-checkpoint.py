import pandas as pd
import numpy as np
from .utils import read_file
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import math 

def preprocess_atac(peak):
    """Binarize the ATAC data for peaks and store in 'binary' layer."""
    binary_matrix = np.where(peak.X.toarray() > 0, 1, 0)
    peak.layers['binary'] = binary_matrix
    return peak

def get_pseudo_cells(adata, k=50):
    """Generate pseudo-cells by averaging nearest neighbors."""
    data = adata.copy()
    df = pd.DataFrame(data.X, index=data.obs.index, columns=data.var.index)
    distances = squareform(pdist(df, metric='hamming'))
    nearest_neighbors_indices = np.argsort(distances, axis=1)[:, 1:k + 1]
    pseudo_df = np.array([df.iloc[nearest_neighbors_indices[i]].sum(axis=0) for i in range(df.shape[0])])
    df += pseudo_df
    df[df > 0] = 1
    return df

def filter_peaks(df, threshold_percentage=0.9):
    """Filter peaks based on the percentage of cells with signal."""
    min_cells = int(threshold_percentage * df.shape[0])
    return df.loc[:, df.sum(axis=0) >= min_cells]

def get_atac(file, peak2gene, cellstates, cicero_file=None, annotation_column='annotation'):
    """Load and preprocess ATAC data, filter by peaks and optionally by Cicero file."""
    atac = read_file(file)
    atac = atac[atac.obs[annotation_column].isin(cellstates)].copy()
    if cicero_file is not None:
        cicero_peaks = pd.read_csv(cicero_file, header=None)[0]
        atac = atac[:, atac.var.index.isin(cicero_peaks)].copy()
    atac = preprocess_atac(atac)
    atac = atac[:, atac.var.index.isin(peak2gene.peakName)].copy()
    return atac

def get_peak_subset(atac, p2g, annotation_column, cellstate, thres_access,
                    use_pseudo_cells=True, todense=True):
    sub = atac[atac.obs[annotation_column] == cellstate]
    if todense == True:
        df = pd.DataFrame(sub.X.todense())
    else:
        df = pd.DataFrame(sub.X)
    df.columns = sub.var.index
    df.index = sub.obs.index
    if use_pseudo_cells == True:
        k = int(math.sqrt(df.shape[0]) * 3)
        df = get_pseudo_cells(df, k=int(math.sqrt(df.shape[0])) * 2)
    pc_df_sub = filter_peaks(df, threshold_percentage=thres_access)
    p2g_sub = p2g[p2g.peakName.isin(list(pc_df_sub.columns))]
    p2g_sub[cellstate] = True
    return p2g_sub

def get_peak2gene(file):
    """Extract and format peak-to-gene mapping from a file."""
    peak2gene = pd.read_csv(file, sep=',')[['peakName', 'geneName']]
    peak2gene['peakName'] = peak2gene['peakName'].str.replace('_', '-')
    peak2gene[['chrom', 'start', 'end']] = peak2gene['peakName'].str.split('-', expand=True)
    peak2gene['start'] = pd.to_numeric(peak2gene['start'])
    peak2gene['end'] = pd.to_numeric(peak2gene['end'])
    return peak2gene

def get_pseudo_cells(df, k=50):
    distances = squareform(pdist(df, metric='hamming'))
    nearest_neighbors_indices = np.argsort(distances, axis=1)[:, 1:k + 1]
    for i in range(df.shape[0]):
        neighbors_sum = df.iloc[nearest_neighbors_indices[i]].sum(axis=0)
        df.iloc[i] += neighbors_sum
    df[df > 0] = 1
    return df

def filter_peaks(df, threshold_percentage=0.9):
    min_cells = int(threshold_percentage * df.shape[0])
    count_above_threshold = df.sum(axis=0)
    filtered_df = df.loc[:, count_above_threshold >= min_cells]
    return filtered_df