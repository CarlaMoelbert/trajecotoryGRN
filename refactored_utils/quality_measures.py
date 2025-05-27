import networkx as nx
import pandas as pd
import numpy as np

def check_interactions(eois, grn, name, weight_column=None):
    """
    Checks if (Source, Target) pairs in eois exist in grn and adds a new column.

    Parameters:
    - eois (pd.DataFrame): The dataframe containing the pairs to check.
    - grn (pd.DataFrame): The dataframe containing the reference pairs.
    - name (str): The name of the new column to be added to eois.
    - weight_column (str, optional): If provided, the function will add the corresponding value 
      from this column in grn instead of True/False.

    Returns:
    - pd.DataFrame: A modified copy of eois with the new column added.
    """
    if weight_column and weight_column in grn.columns:
        grn_dict = {(row['Source'], row['Target']): row[weight_column] for _, row in grn.iterrows()}
    else:
        grn_dict = {(row['Source'], row['Target']): True for _, row in grn.iterrows()}
    eois[name] = eois.apply(lambda row: grn_dict.get((row['Source'], row['Target']), False), axis=1)
    return eois

def count_edges_of_interest(df, eois, with_direction=False):
    count_eoi_found = 0
    if with_direction == True:
        df['Direction'] = df['Coef'].apply(lambda x: -1 if x < 0 else 1 if x > 0 else 0)
    for index, eoi in eois.iterrows():
        if with_direction == False:
            x = df[(df.Source == eoi['Source']) & (df.Target == eoi['Target'])]
        else:
            x = df[(df.Source == eoi['Source']) & (df.Target == eoi['Target']) & (df.Direction == eoi['Direction'])]
        if x.shape[0] > 0:
            count_eoi_found += 1
    return count_eoi_found

def get_centralities(edges, nodes):
    G = nx.DiGraph()
    G.add_edges_from(edges[['Source', 'Target']].values)
    centralities = {'Betweenness': nx.betweenness_centrality(G, weight='Weight'), 'Closeness': nx.closeness_centrality(G, distance='Coef'), 'Eigenvector': nx.eigenvector_centrality(G, max_iter=1000, weight='Coef'), 'In-Degree': dict(G.in_degree(weight=None)), 'Out-Degree': dict(G.out_degree(weight=None))}
    genes_in_all = reduce(lambda a, b: a & b, (set(d.keys()) for d in centralities.values()))
    nodes = nodes[nodes['gene'].isin(genes_in_all)].copy()
    for col_name, centrality_dict in centralities.items():
        nodes[col_name] = nodes['gene'].map(centrality_dict)
    nodes['In-Degree'] = nodes['In-Degree'] / nodes.shape[0]
    nodes['Out-Degree'] = nodes['Out-Degree'] / nodes.shape[0]
    return nodes

def multi_compare_plot(df1, df2, compare_cols, output, id_col='gene', label_top_n=5, title_prefix='Comparison', name2='Cell State Specific', name1='Overall'):
    n = len(compare_cols)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    axes = axes.flatten()
    for i, compare_col in enumerate(compare_cols):
        ax = axes[i]
        merged = pd.merge(df1[[id_col, compare_col]], df2[[id_col, compare_col]], on=id_col, how='outer', suffixes=('_df1', '_df2')).fillna(0)
        x = merged[f'{compare_col}_df1']
        y = merged[f'{compare_col}_df2']
        merged['dist_to_diag'] = np.abs(x - y) / np.sqrt(2)
        top_outliers = merged.nlargest(label_top_n, 'dist_to_diag')
        ax.scatter(x, y, alpha=0.6, color='grey')
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray')
        texts = [ax.text(row[f'{compare_col}_df1'], row[f'{compare_col}_df2'], row[id_col], fontsize=8) for _, row in top_outliers.iterrows()]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5), force_points=0.3, expand_points=(1.2, 1.4), only_move={'points': 'y', 'text': 'xy'})
        ax.set_xlabel(f'{name1}', fontsize=8)
        ax.set_ylabel(f'{name2}', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_title(f'{title_prefix}: {compare_col}', fontsize=10)
        ax.grid(True)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    fgt.save_or_show_plot(output)