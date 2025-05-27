import networkx as nx
import numpy as np
import pandas as pd
import scipy
from refactored_utils.utils import *
from refactored_utils.visualizations import *

def filter_nodes(df_nodes, df_edges):
    tfactivity_cols = [col for col in df_nodes.columns if col.endswith('_TFactivity')]
    df_nodes['Max_TFactivity'] = df_nodes[tfactivity_cols].max(axis=1)
    df_nodes_sorted = df_nodes.sort_values(by='Max_TFactivity', ascending=True).copy()
    G = nx.from_pandas_edgelist(df_edges, source='Source', target='Target', create_using=nx.DiGraph)
    for _, row in df_nodes_sorted.iterrows():
        node = row['gene']
        max_activity = row['Max_TFactivity']
        if max_activity > 0:
            continue
        if node in G:
            G_temp = G.copy()
            G_temp.remove_node(node)
            if nx.is_weakly_connected(G_temp):
                G.remove_node(node)
    remaining_nodes = list(G.nodes)
    df_nodes_filtered = df_nodes[df_nodes['gene'].isin(remaining_nodes)].copy()
    df_edges_filtered = df_edges[df_edges['Source'].isin(remaining_nodes) & df_edges['Target'].isin(remaining_nodes)].copy()
    return (df_nodes_filtered, df_edges_filtered)

def filter_basenetwork(grn, expr, annotation, layer, gois, keep_selfregulation=False, weight_cutoff=None, filter_coexpr=0):
    if not keep_selfregulation:
        grn = grn[grn.Source != grn.Target]
    degs = annotation.loc[annotation.DEG == True, 'gene']
    grn = grn[grn.Target.isin(set(degs).union(set(grn.Source)))]
    genes = list(set(grn.Source).union(set(grn.Target)).intersection(expr.var.index))
    grn = grn[grn.Source.isin(genes) & grn.Target.isin(genes)]
    genes = list(set(grn.Source).union(set(grn.Target)).intersection(expr.var.index))
    if not expr.var_names.is_unique:
        expr = expr[:, ~expr.var_names.duplicated()].copy()
    expr = expr[:, genes]
    expr_matrix = get_expr_matrix(expr, todense=False, layer=layer)
    grn['co_expression'] = grn.apply(lambda row: compute_coexpression(expr_matrix, row['Source'], row['Target']), axis=1)
    if filter_coexpr != 0:
        grn = grn[grn.co_expression > filter_coexpr]
    grn = largest_weakly_connected_component(grn)
    expr_matrix = expr_matrix[genes]
    return (grn, expr_matrix)

def largest_weakly_connected_component(edges_df, source_col='Source', target_col='Target'):
    """Reduce edges dataframe to the largest weakly connected component."""
    G = nx.from_pandas_edgelist(edges_df, source=source_col, target=target_col, create_using=nx.DiGraph())
    largest_component = max(nx.weakly_connected_components(G), key=len)
    return edges_df[edges_df[source_col].isin(largest_component) & edges_df[target_col].isin(largest_component)]

def remove_uninformative_genes(grn, degs=None):
    if degs is not None:
        grn = grn[grn.Target.isin(set(degs).union(set(grn.Source)))]
    grn = largest_weakly_connected_component(grn)
    return grn

def get_degs(data, cells_in_state, cells_in_other_states, threshold=0.05, max_features=20, name='', output=None, gois=None, saveFigures=False, folder='figures'):
    data_matrix = data.copy()
    feature_pvals = {}
    for feature in data_matrix.columns:
        group1 = data_matrix.loc[cells_in_state, feature].dropna()
        group2 = data_matrix.loc[cells_in_other_states, feature].dropna()
        if len(group1) < 3 or len(group2) < 3:
            feature_pvals[feature] = np.nan
            continue
        _, pval = scipy.stats.ranksums(group1, group2, alternative='greater')
        feature_pvals[feature] = pval
    feature_pvals_df = pd.DataFrame(list(feature_pvals.items()), columns=['feature', 'pval'])
    feature_fc = {feature: np.log2((data_matrix.loc[cells_in_state, feature].mean() + 1e-10) / (data_matrix.loc[cells_in_other_states, feature].mean() + 1e-10)) for feature in data_matrix.columns}
    feature_fc_df = pd.DataFrame(list(feature_fc.items()), columns=['feature', 'log2FC'])
    volcano_df = feature_pvals_df.merge(feature_fc_df, on='feature')
    volcano_df['neg_log10_pval'] = -np.log10(volcano_df['pval'].replace(0, 1e-300))
    volcano_df = volcano_df.replace([np.inf, -np.inf], np.nan).dropna()
    if volcano_df.shape[0] == 0:
        print('Warning: No valid features to plot in volcano plot.')
        return
    if (gois != None) & (saveFigures == True):
        volcano_plot(volcano_df, output=f'{folder}/{output}_{name}_volcano.png', gois=gois, h=5, w=5)
    selected_features = feature_pvals_df[feature_pvals_df['pval'] < threshold]
    selected_features = selected_features.sort_values(by=['pval', 'feature'])
    selected_features = selected_features.reset_index()
    if len(selected_features) > max_features:
        selected_features = selected_features.head(max_features)
    return selected_features['feature'].tolist()