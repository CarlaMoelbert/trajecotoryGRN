import numpy as np
import seaborn as sns
import pandas as pd
import scanpy as sc
from pySankey.sankey import sankey
from upsetplot import UpSet
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

# ========== Utility Functions ========== #
def save_or_show_plot(output=None):
    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".svg", ".pdf"}

    if output:
        file_ext = os.path.splitext(output)[1].lower()  # Extract extension
        if file_ext not in valid_extensions:
            print(f"⚠️ Warning: '{output}' has no valid extension. Saving as .png instead.")
            output += ".png"  # Default to PNG if no valid extension
        
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"📸 Plot saved as: {output}")
        plt.close()  # Close the plot to free memory
    else:
        plt.show()  # Show plot when no output is given

# ========== Heatmaps ========== #
def get_heatmap(df, suffix="_expression", check_col="DEG", sort_col="rank_id",
                h=3, w=8, output=None):
    df = df[df[check_col] == True].sort_values(sort_col)
    rank_cmap = sns.color_palette("viridis", as_cmap=True)
    rank_series = df.set_index("gene")[sort_col]
    rank_norm = plt.Normalize(rank_series.min(), rank_series.max())
    rank_colors = rank_cmap(rank_norm(rank_series))

    data = df[[col for col in df.columns if col.endswith(suffix)]].T
    cmap = "vlag" if data.min().min() < 0 else None
    center = 0 if data.min().min() < 0 else None

    # Increase the height of the col_colors row
    g = sns.clustermap(data, figsize=(w, h),  # Add extra height to prevent overlap
                       col_cluster=False, row_cluster=False,
                       col_colors=np.array([rank_colors] * 5),  # Expand col_colors for thickness
                       cmap=cmap, center=center,
                       cbar_kws={"orientation": "horizontal",
                                 "shrink": 0.5,
                                 "aspect": 20},
                       tree_kws={"linewidths": 0},  # Remove unwanted tree space
                       dendrogram_ratio=(0.05, 0.2), xticklabels=True)  

    # Remove the suffix from y-axis labels
    new_labels = [label.get_text().replace(suffix, "") for label in g.ax_heatmap.get_yticklabels()]
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 8)

    g.ax_heatmap.set_yticklabels(new_labels)

    # Adjust the color bar position to prevent overlap
    g.cax.set_position([0.2, 0.92, 0.2, 0.02])  # [left, bottom, width, height]
    if suffix == "_expression":
        g.cax.set_title("Expression", fontsize=10)
    elif suffix == "_TFactivity":
        g.cax.set_title("TF Activity", fontsize=10)
    else:
        g.cax.set_title(suffix, fontsize=10)

    save_or_show_plot(output)


# ========== Sankey Plot ========== #
def get_sankey_plot(edges, nodes, output=None):
    """
    Generates a Sankey diagram using pySankey.
    - Uses rank information from nodes to structure the diagram.
    - Ensures correct flow visualization.
    - Colors each rank using the "viridis" colormap.
    """
    # Merge rank information from nodes
    edges = edges.merge(nodes.rename(columns={"gene": "Source",
                                              "rank_id": "rank_Source"}),
                        on="Source", how="left")
    edges = edges.merge(nodes.rename(columns={"gene": "Target",
                                              "rank_id": "rank_Target"}),
                        on="Target", how="left")

    # Convert rank IDs to string
    edges["rank_Source"] = edges["rank_Source"].astype(str)
    edges["rank_Target"] = edges["rank_Target"].astype(str)

    # Sort edges based on rank
    edges = edges.sort_values(by=["rank_Source", "rank_Target"], ascending=False)

    # Get unique ranks for coloring
    unique_ranks = sorted(set(edges["rank_Source"]).union(set(edges["rank_Target"])))

    # Create a colormap from viridis
    norm = mcolors.Normalize(vmin=0, vmax=len(unique_ranks) - 1)
    cmap = plt.get_cmap("viridis")
    rank_colors = {rank: cmap(norm(i)) for i, rank in enumerate(unique_ranks)}

    # Create color mapping dictionary for Sankey
    colorDict = {rank: rank_colors[rank] for rank in unique_ranks}

    # Generate Sankey plot with correct arguments
    sankey(
        left=edges["rank_Source"],  # Source (now strings)
        right=edges["rank_Target"],  # Target (now strings)
        fontsize=10,
        colorDict=colorDict
    )

    save_or_show_plot(output)


    
# ========== DEG Selection Heatmap ========== #
def show_deg_selection(df, output=None):
    """
    Shows a heatmap for DEG selection.
    - Moves the color bar to the center-top.
    - Annotates each cell with its placement value.
    - Prevents x-axis labels from overlapping with the heatmap or color bar.
    - Removes axis titles.
    """

    df = df[df["pval"] < 0.05]
    heatmap_data = df.pivot(index="feature", columns="comparison", values="pval")
    placement_data = df.pivot(index="feature", columns="comparison", values="placement")

    # Create the heatmap with annotations
    g = sns.clustermap(heatmap_data, figsize=(5, 5),
                       col_cluster=False, row_cluster=False,
                       annot=placement_data, fmt=".0f",
                       annot_kws={"size": 10, "color": "white", "weight": "bold"},
                       linewidths=0.5, linecolor='black',
                       cbar_kws={"orientation": "horizontal",
                                 "shrink": 0.5, "aspect": 30},
                       xticklabels=True)

    # Move the color bar to the center-top
    g.cax.set_position([0.3, 0.9, 0.4, 0.04])  # [left, bottom, width, height]
    g.cax.set_title("P-Value", fontsize=10)
    # Adjust x-axis labels to prevent overlap
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Remove axis labels
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    save_or_show_plot(output)

# ========== UpSet Plot ========== #
def show_peak_upset(data_dict, width=10, height=5, output=None):
    """Generates an UpSet plot to visualize peak overlaps."""
    binary_df = pd.DataFrame({key: [el in val for el in sorted(set().union(*data_dict.values()))]
                              for key, val in data_dict.items()})
    fig = plt.figure(figsize=(width, height))
    df = binary_df.set_index(list(binary_df.columns))
    UpSet(df, show_counts='%d', sort_by=None).plot(fig=fig)
    
    save_or_show_plot(output)

# ========== Bar Plot for Peaks ========== #
def visualize_peaks(p2g_filtered, p2g_og, gois, output=None):
    """Visualizes peaks in a bar plot."""
    
    df = pd.DataFrame({
        'Total': p2g_og[p2g_og.geneName.isin(gois)].geneName.value_counts(),
        'Filtered': p2g_filtered[p2g_filtered.geneName.isin(gois)].geneName.value_counts()
    }).fillna(0).sort_index()

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(df.index, df['Filtered'], color='black', label='Peaks Kept')
    ax.bar(df.index, df['Total'] - df['Filtered'], bottom=df['Filtered'], 
           color='lightgray', label='All Peaks')

    ax.set_ylabel('Number Of Peaks')
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.legend()

    save_or_show_plot(output)

# ========== Adjacency Heatmap ========== #
def get_adjacency(nodes_df, edges_df, gois=None, w=10, h=10, s=12, only_DATF=False, output=None, axis_labels=True):
    if only_DATF:
        nodes_df = nodes_df[nodes_df["DATF"]]

    genes_sorted = nodes_df.sort_values("rank_id")["gene"].unique()
    edges_df = edges_df[edges_df["Source"].isin(genes_sorted) & edges_df["Target"].isin(genes_sorted)]

    if "Coef" not in edges_df.columns:
        edges_df["Coef"] = 1

        
    pivot = edges_df.pivot(index="Source",
                           columns="Target",
                           values="Coef").reindex(index=genes_sorted,
                                                  columns=genes_sorted).fillna(0)

    # Debugging: Check if pivot has labels
    print("Pivot Index:", pivot.index)
    print("Pivot Columns:", pivot.columns)

    # Filter if GOIs are provided
    if gois is not None:
        pivot = pivot.reindex(index=gois, columns=gois, fill_value=0)

    # Generate rank-based color annotations
    rank_cmap = sns.color_palette("viridis", as_cmap=True)
    rank_norm = plt.Normalize(nodes_df["rank_id"].min(), nodes_df["rank_id"].max())
    rank_colors = nodes_df.set_index("gene")["rank_id"].map(lambda v: rank_cmap(rank_norm(v))).tolist()
    row_colors = np.array(rank_colors)[:, :3]  
    col_colors = row_colors.copy()

    # Create the heatmap
    g = sns.clustermap(pivot, figsize=(w, h),
                       col_cluster=False, row_cluster=False,
                       row_colors=row_colors, col_colors=col_colors,
                       cmap="RdBu_r",
                       colors_ratio=0.01, square=False, center=0, vmin=-1, vmax=1,
                       linewidths=0.0,
                       linecolor="gray",
                       yticklabels=True, xticklabels=True)

    # Ensure axis labels are displayed
    if axis_labels:
        g.ax_heatmap.set_xticks(np.arange(len(pivot.columns)))
        g.ax_heatmap.set_xticklabels(pivot.columns, rotation=90, fontsize=8)

        g.ax_heatmap.set_yticks(np.arange(len(pivot.index)))
        g.ax_heatmap.set_yticklabels(pivot.index, fontsize=8)
    else:
        g.ax_heatmap.set_xticklabels([])
        g.ax_heatmap.set_yticklabels([])

    # Align the color bars
    g.cax.set_position([0.1, 0.5, 0.03, 0.1])
    g.cax.set_title("Edge Weight", fontsize=s)
    g.cax.tick_params(labelsize=s - 2)

    # Add "Rank ID" legend
    legend_ax = g.fig.add_axes([0.1, 0.3, 0.03, 0.1])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=rank_norm)
    cbar = g.fig.colorbar(sm, cax=legend_ax, orientation="vertical")
    cbar.ax.set_title("Rank ID", fontsize=s, pad=10)
    cbar.ax.tick_params(labelsize=s - 2)

    # Save or display the plot
    save_or_show_plot(output)


# ========== Compare Network Types ========== #
def compare_networktypes(df, h=3, w=15, output=None,
                         parameters=['dataset', 'fpr', 'thres_access', 'max_features', 'max_tf', "weight_cutoff"],
                         measures=['Modularity', 'Clustering_Coefficient', 'Network_Density',
                                   'Robustness', 'Average_Path_Length', 'nr_found']):
    """
    Compares 'overall' and 'cellstate' network types across different measures.
    - Colors points based on dataset.
    - Uses lower alpha to prevent overplotting.
    - Adds y=x reference line.
    """
    sns.set_style("whitegrid")

    # Prepare data for comparison between "overall" and "cellstate"
    comparison_data = []

    for _, group in df.groupby(parameters):
        if {'overall', 'cellstate'}.issubset(set(group['networktype'])):
            overall_row = group[group['networktype'] == 'overall']
            cellstate_row = group[group['networktype'] == 'cellstate']

            row = {'dataset': group['dataset'].iloc[0]}  # Store dataset
            for measure in measures:
                row[f'{measure}_overall'] = overall_row[measure].values[0]
                row[f'{measure}_cellstate'] = cellstate_row[measure].values[0]

            comparison_data.append(row)

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    if comparison_df.empty:
        print("No matching pairs found.")
        return

    # Ensure measure columns are numeric
    for measure in measures:
        comparison_df[f'{measure}_overall'] = pd.to_numeric(comparison_df[f'{measure}_overall'], errors='coerce')
        comparison_df[f'{measure}_cellstate'] = pd.to_numeric(comparison_df[f'{measure}_cellstate'], errors='coerce')

    # Generate color palette for datasets (same as `benchmark_parameter`)
    datasets = comparison_df["dataset"].unique()
    dataset_palette = dict(zip(datasets, sns.color_palette("tab10", len(datasets))))

    # Create subplots for each measure
    fig, axes = plt.subplots(1, len(measures), figsize=(w, h), sharex=False, sharey=False)

    for i, measure in enumerate(measures):
        ax = axes[i]

        # Scatter plot for each dataset
        for dataset, group_df in comparison_df.groupby("dataset"):
            ax.scatter(group_df[f'{measure}_overall'], group_df[f'{measure}_cellstate'],
                       color=dataset_palette[dataset], alpha=0.5, label=dataset)  # Lower alpha for better visibility

        # Compute valid min/max values for reference line
        overall_vals = comparison_df[f'{measure}_overall'].dropna()
        cellstate_vals = comparison_df[f'{measure}_cellstate'].dropna()

        if not overall_vals.empty and not cellstate_vals.empty:
            min_val = min(overall_vals.min(), cellstate_vals.min())
            max_val = max(overall_vals.max(), cellstate_vals.max())

            ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='grey')

        ax.set_title(measure)
        ax.set_xlabel('Overall')
        ax.set_ylabel('Cellstate')

    # Add shared legend for dataset colors
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=dataset_palette[ds], label=ds) for ds in datasets]
    fig.legend(handles=handles, loc="upper center", ncol=len(datasets)//2 + 1, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_or_show_plot(output)

# ========== Benchmark Parameter ========== #
def benchmark_parameter(df, parameter, w=20, h=5, output=None,
                        measures=['Modularity', 'Clustering_Coefficient', 'Network_Density',
                                  'Robustness', 'Average_Path_Length', 'nr_found'],
                        measure_ranges=None,  # Optional dictionary for y-axis limits
                        **defaults):
    """
    Plots scatter comparisons of different measures against a given parameter.
    - Colors points based on dataset.
    - Uses point shape and line type based on networktype.
    - Connects points within groups while preventing loops.
    - Adds a vertical dotted grey line if the default value of the parameter is given.
    - Allows specifying y-axis ranges for individual measures.
    - Moves legend above plots to prevent overlap.
    - Formats X-axis float values to show at most 2 decimal places.
    """
    sns.set_style("whitegrid")

    # Filter DataFrame to match default parameters except for the target parameter
    filtered_df = df.copy()
    for param, default in defaults.items():
        if param != parameter:
            filtered_df = filtered_df[filtered_df[param] == default]

    # Handle case where no data matches the filter
    if filtered_df.empty:
        print("No data matches the specified default parameter settings.")
        return
            
    # Generate unique colors for each dataset
    datasets = filtered_df["dataset"].unique()
    dataset_palette = dict(zip(datasets, sns.color_palette("tab10", len(datasets))))

    # Define marker styles & line styles for network types
    network_types = filtered_df["networktype"].unique()
    multiple_network_types = len(network_types) > 1  # Check if we need to add it to legend

    markers = ["o", "s", "D", "^", "v", "P", "*", "X"][:len(network_types)]
    linestyles = ["-", "--", "-.", ":"][:len(network_types)]
    network_style = {nt: {"marker": markers[i],
                          "linestyle": linestyles[i]} for i, nt in enumerate(network_types)}

    # Default value for the parameter
    default_value = defaults.get(parameter, None)

    # Create subplots
    fig, axes = plt.subplots(1, len(measures), figsize=(w, h), sharex=False, sharey=False)

    for i, measure in enumerate(measures):
        ax = axes[i]

        # Plot each dataset-networktype combination separately
        for (dataset, networktype), group_df in filtered_df.groupby(["dataset", "networktype"]):
            style = network_style[networktype]
            color = dataset_palette[dataset]

            # Ensure points are connected in order but prevent looping
            group_df = group_df.sort_values(parameter)

            ax.plot(group_df[parameter], group_df[measure], marker=style["marker"], linestyle=style["linestyle"],
                    color=color, alpha=0.8, label=f"{dataset} ({networktype})" if multiple_network_types else dataset)

        # Set title and labels
        ax.set_title(measure)
        ax.set_xlabel(parameter)

        # Apply user-specified y-axis range if provided
        if measure_ranges and measure in measure_ranges:
            ax.set_ylim(measure_ranges[measure])

        # Add vertical line at default value if provided
        if default_value is not None:
            ax.axvline(default_value, linestyle="dotted", color="grey", alpha=0.8, label=f"Default: {default_value}")


        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}".rstrip("0").rstrip(".")))

    # Shared Y-axis label on the first plot
    axes[0].set_ylabel("Measure Value")

    # Build legend dynamically
    handles = [plt.Line2D([0], [0], color=dataset_palette[ds], linestyle="-", label=ds) for ds in datasets]

    if multiple_network_types:
        handles += [plt.Line2D([0], [0], color="black", marker=network_style[nt]["marker"],
                               linestyle=network_style[nt]["linestyle"], label=nt) for nt in network_types]

    fig.legend(handles=handles, loc="upper center", ncol=max(len(datasets), len(network_types)) if multiple_network_types else len(datasets), 
               fontsize=10, bbox_to_anchor=(0.5, 1.3))

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 1.05])  
    # Save or show the plot
    save_or_show_plot(output)


# ========== Top Centralities Plot ========== #
def plot_top_centralities(edges, output=None):
    """
    Plots the top 10 nodes for different centrality measures.
    - Scales In-Degree and Out-Degree centralities between 0 and 1.
    - Sets x-axis range from 0 to 1 for all centralities.
    """
    sns.set_style("whitegrid")

    # Create directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges[["Source", "Target"]].values)

    # Compute centralities
    centralities = {
        "Betweenness": nx.betweenness_centrality(G),
        "Closeness": nx.closeness_centrality(G),
        #"Eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
        "In-Degree": dict(G.in_degree(weight=None)),
        "Out-Degree": dict(G.out_degree(weight=None))
    }

    # Scale In-Degree and Out-Degree between 0 and 1
    for key in ["In-Degree", "Out-Degree"]:
        max_value = max(centralities[key].values(), default=1)  # Avoid division by zero
        centralities[key] = {k: v / max_value for k, v in centralities[key].items()}

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    axes = axes.flatten()

    for i, (name, centrality) in enumerate(centralities.items()):
        ax = axes[i]

        # Get top 10 nodes by centrality
        top_10 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        nodes, values = zip(*top_10) if top_10 else ([], [])

        # Create bar plot
        bars = ax.barh(range(len(nodes)), values, color="black", edgecolor="black")
        #ax.set_xlim(0, 1)  # Set x-axis range to 0-1

        # Set axis labels
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes)
        ax.invert_yaxis()
        ax.set_xlabel(f"{name} Centrality")

    plt.tight_layout()
    save_or_show_plot(output)

# ========== GOI Centralities Plot ========== #
def plot_goi_centralities(edges, gois, output=None, w= 10, h=3):

    sns.set_style("whitegrid")

    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges[["Source", "Target"]].values)

    # Compute centralities
    centralities = {
        "Betweenness": nx.betweenness_centrality(G),
        "Closeness": nx.closeness_centrality(G),
        #"Eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
        "In-Degree": dict(G.in_degree(weight=None)),
        "Out-Degree": dict(G.out_degree(weight=None))
    }

    # Scale In-Degree and Out-Degree between 0 and 1
    for key in ["In-Degree", "Out-Degree"]:
        max_value = max(centralities[key].values(), default=1)  # Avoid division by zero
        centralities[key] = {k: v / max_value for k, v in centralities[key].items()}

    fig, axes = plt.subplots(1, 4, figsize=(w, h))
    axes = axes.flatten()

    for i, (name, centrality) in enumerate(centralities.items()):
        ax = axes[i]

        # Sort all genes by centrality value (highest first)
        sorted_genes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        overall_ranks = {gene: rank + 1 for rank, (gene, _) in enumerate(sorted_genes)}

        # Filter for GOIs and sort by centrality
        goi_centralities = {gene: centrality.get(gene, 0) for gene in gois}
        sorted_gois = sorted(goi_centralities.items(), key=lambda x: x[1], reverse=True)
        nodes, values = zip(*sorted_gois) if sorted_gois else ([], [])

        # Create bar plot
        bars = ax.barh(range(len(nodes)), values, color="black", edgecolor="black")

        # Annotate bars with overall rank or "Not Found"
        for bar, node in zip(bars, nodes):
            rank_text = f"#{overall_ranks[node]}" if node in overall_ranks else "Not Found"
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, rank_text, 
                    va='center', ha='left', fontsize=10, color="black", fontweight="bold")

        # Set axis labels and x-axis range
        #ax.set_xlim(0, 1)
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes)
        ax.invert_yaxis()  # Highest values at the top
        ax.set_xlabel(f"{name} Centrality")
        ax.set_ylabel("GOI")

    plt.tight_layout()
    save_or_show_plot(output)


# ========== General Heatmap Utility ========== #
def plot_heatmap(data, ax, cmap='RdBu_r', center=0):
    """Plots a heatmap with consistent styling."""
    sns.heatmap(data, cmap=cmap, center=center, linewidths=0.5, linecolor='gray', square=True, ax=ax)



def show_distribution(df, column, name, title, counts=True):
    output = f"{name}_{title}_{column}.png"

    # Histogram of a specific column in dataframe df 
    # The design should fit to the already existing figures
    if counts == True:
        values = df[column].value_counts()
        xlabel = f"Number of Occurences of each {column}"
    else:
        values = df[column]
        xlabel = column

    fig, ax = plt.subplots(figsize=(3, 3))
    if min(values) < 0:
        plt.hist(values[values > 0], color="blue", bins=20)
        plt.hist(values[values < 0], color="red", bins=20)
        
    else:
        plt.hist(values, color="black", bins=20)
    plt.xlabel(xlabel) 
    plt.ylabel("Frequency")  
    
    save_or_show_plot(output)



# Compute co-expression for a gene pair
def compute_coexpression(expr_matrix, gene1, gene2):
    # Get expression status (1 if expressed, 0 if not)
    expressed_gene1 = expr_matrix[gene1] > 0
    expressed_gene2 = expr_matrix[gene2] > 0
    
    # Count number of cells expressing both genes
    coexpressed_cells = (expressed_gene1 & expressed_gene2).sum()
    
    # Total number of cells
    total_cells = expr_matrix.shape[0]
    
    return coexpressed_cells / total_cells if total_cells > 0 else 0

def compare_coexpression(expr, grn, output):
    # Get all unique gene pairs in the dataset

    
    all_genes = set(expr.columns).intersection(set(grn.Source).union(set(grn.Target)))
    all_pairs = set(combinations(all_genes, 2))
    
    # Get connected pairs from GRN
    connected_pairs = set(zip(grn["Source"], grn["Target"]))
    
    # Get non-connected pairs (randomly sampled from all possible pairs)
    non_connected_pairs = list(all_pairs - connected_pairs)
    
    # Compute co-expression values
    coexpression_data = []
    
    for gene1, gene2 in connected_pairs:
        coexpression_data.append({"Coexpression": compute_coexpression(expr, gene1, gene2), "Group": "Connected"})
    
    for gene1, gene2 in non_connected_pairs:
        coexpression_data.append({"Coexpression": compute_coexpression(expr, gene1, gene2), "Group": "Not Connected"})
    
    # Convert to DataFrame
    coexpression_df = pd.DataFrame(coexpression_data)
    
    # Plot violin plot
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="Group", y="Coexpression", data=coexpression_df,
                   inner="box", fill = False)
    #plt.xlabel("Gene Pair Group")
    plt.ylabel("Co-expression Fraction")
    plt.title("Distribution of Co-expression in Connected vs. Non-Connected Gene Pairs")

    save_or_show_plot(output)

def volcano_plot(df, output=None, gois=None, h=5, w=5):
    # Compute -log10(p-value) if not already in the DataFrame
    if 'neg_log10_pval' not in df.columns:
        df['neg_log10_pval'] = -np.log10(df['pval'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['neg_log10_pval', 'log2FC']) 
 
    plt.figure(figsize=(w, h))

    # Scatter plot: default gray color
    plt.scatter(df['log2FC'], df['neg_log10_pval'], color='gray',
                alpha=0.7, label='All genes')

    # Highlight genes of interest if provided
    if gois:
        df_gois = df[df['feature'].isin(gois)]
        plt.scatter(df_gois['log2FC'], df_gois['neg_log10_pval'],
                    color='red', label='GOIs')

        # Add labels for genes of interest with slight y-offsets to avoid overlap
        for i, row in df_gois.iterrows():
            y_offset = 0.1 if i % 2 == 0 else -0.1  # Alternate label positions
            plt.text(row['log2FC'], row['neg_log10_pval'] + y_offset, row['feature'],
                     fontsize=12, ha='right', bbox=dict(facecolor='white', alpha=0.5,
                                                        edgecolor='none'))

    # Labels and title
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 p-value')
    plt.title('Volcano Plot')

    # Add threshold lines
    plt.axhline(y=-np.log10(0.05), color='blue', linestyle='--',
                label='p=0.05 threshold')
    plt.axvline(x=0, color='black', linestyle='--')

    plt.legend()
    
    # Save or display the plot
    save_or_show_plot(output)
