import refactored_utils.visualizations as visual
from refactored_utils.benchmark import *
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import numpy as np

print("Start")
parameters = [ "degs", "thres_access", "fpr", "edge_cutoff_expr", "max_tf"]
measure_lists =[['genes_found', "edges_found", "edges_correctly_identified"],
                ["nodes",  "edges", "Modularity", 'Network_Density']]


df_expr = pd.read_csv("data/benchmark_summary_expressionbasedWeights.csv")
df_activity = pd.read_csv("data/benchmark_summary_TFactivitybasedWeights.csv")

fpr_default = 0.02,
thres_access_default = 0.75,
max_features_default = 25,
max_tf_default = 50,
edges_cutoff_expr_default=2000,
edges_cutoff_act_default=500

for parameter in parameters:
    for networktype in ["overall", "cellstate"]:
        for measures in measure_lists:
            visual.benchmark_parameter(
                df_expr[df_expr.networktype == networktype], h =3,w = 15,
                measures = measures,
                parameter=parameter,
                fpr=fpr_default,
                thres_access=thres_access_default,
                degs=max_features_default,
                max_tf=max_tf_default,
                edge_cutoff_expr=edges_cutoff_expr_default,
                measure_ranges={"Modularity": (0, 0.1),
                                "Network_Density": (0, 0.6),
                                "genes_found":(0,31),
                                "edges_found":(0,76),
                                "edges_correctly_identified":(0,76)},
                output=f"output/benchmark_{networktype}_{parameter}_{measures[0]}.png"
            )
print("_______________________________________________________________________")

id_cols = ["dataset", "degs", "thres_access", "fpr", "edge_cutoff_expr", "max_tf"]
parameter_cols = ["edge_cutoff_act", "edges", "nodes", "genes_found", "edges_found", "edges_correctly_identified", "Modularity", "Network_Density"]
print(df_expr.networktype.value_counts())

df_overall   = filter_df(df_expr[df_expr.networktype == "overall"])
df_cellstate = filter_df(df_expr[df_expr.networktype == 'cellstate'])

df_type = merge_dataframes_with_prefix(df_overall, df_cellstate, id_cols,
                             "Overall Network", "Cellstate-Specific Network" )

print(f"overall: {df_overall.shape} cellstate: {df_cellstate.shape} both: {df_type.shape}")
compare_networktypes(df_type,
                     type1 = "Overall Network",
                     type2="Cellstate-Specific Network", 
                     output=f"output/comparison_networktype.png")


df_weight = merge_dataframes_with_prefix(df_expr, df_activity, id_cols,
                             "Expression", "TF Activity" )
compare_networktypes(df_weight,
                     type1 = "Expression",
                     type2="TF Activity", output=f"output/comparison_weightTypes.png")



print("_______________________________________________________________________")

def get_heatmap_comparison(df_full, dataset, output,
                           measures = ["genes_found", "edges_found", "edges_correctly_identified"]):
    print(df_full)
    df = df_full[df_full.dataset == dataset]
    print(df.shape)
    df = df[df.measure.isin(["genes_found", "edges_found", "edges_correctly_identified"])]
    print(df.shape)
    num_measures = len(measures)


    # Dictionary to store heatmap data for each measure
    heatmap_dict = {
        measure: df[df['measure'] == measure]
        .pivot_table(index='value_cellstate', columns='value_overall',
                     aggfunc='size', fill_value=0)
        for measure in measures
    }

    fig, axes = plt.subplots(1, num_measures, figsize=(6 * num_measures, 5), squeeze=False)
    
    for i, measure in enumerate(measures):
        data = heatmap_dict[measure]
        # Create annotation labels only for non-zero cells
        annot = data.where(data != 0).astype(str).replace('nan', '')
        
        ax = axes[0, i]
        sns.heatmap(data, annot=annot, fmt='', cmap='viridis', ax=ax)
        ax.set_xlabel(f'{measure} for Overall Network')
        ax.set_ylabel(f'{measure} for Cellstate-Specific Network')
    
    plt.tight_layout()
    visual.save_or_show_plot(output)


measures = ['genes_found', "edges_found", "edges_correctly_identified"]
m =  measures+ ["networktype"]
dfs = {}

df_expr_sub = df_expr[["dataset", "networktype","degs", "thres_access", "fpr",
                       "edge_cutoff_expr", "max_tf",
                       "genes_found", "edges_found", "edges_correctly_identified"]]

for networktype in ["overall", "cellstate"]:
    df_1 = df_expr_sub[df_expr_sub.networktype == networktype]
    df_melted = df_1.melt(id_vars=[col for col in df_1.columns if col not in m], 
                     value_vars=measures, 
                     var_name='measure', 
                     value_name=f'value_{networktype}')
    print(f"{networktype}: {df_melted.shape}")
    dfs[networktype] = df_melted


print(dfs["overall"].columns)
print(dfs["cellstate"].columns)

df_comp = pd.merge(dfs["overall"], dfs["cellstate"] )




get_heatmap_comparison(df_comp, "Eryth", "output/Overall_Cellstate_Found_Eryth.png")
get_heatmap_comparison(df_comp, "Mono", "output/Overall_Cellstate_Found_Mono.png")


print("_______________________________________________________________________")

def plot_fraction_differences_nz(df, x_param, param_name, default_params, width=10, height=3):
    import matplotlib.pyplot as plt
    import seaborn as sns

    comparisons = [
        ("diagonal_window", "upper_triangle"),
        ("diagonal_window", "lower_triangle"),
        ("upper_triangle", "lower_triangle"),
    ]
    titles = [
        "Diagonal vs Upper Triangle",
        "Diagonal vs Lower Triangle",       
        "Upper vs Lower Triangle",
    ]

    # Filter by fixed params except x_param
    filtered_df = df.copy()
    for key, val in default_params.items():
        if key != x_param:
            filtered_df = filtered_df[filtered_df[key] == val]
            

    datasets = filtered_df["dataset"].unique()
    colors = sns.color_palette("tab10", n_colors=len(datasets))
    color_dict = dict(zip(datasets, colors))

    fig, axs = plt.subplots(1, 3, figsize=(width, height))
    
    for ax, (s1, s2), title in zip(axs, comparisons, titles):
        for dataset in datasets:
            df_subset = filtered_df[filtered_df["dataset"] == dataset]

            pos_pivot = df_subset.pivot(index=x_param, columns="section", values="positive_fraction_nz")
            neg_pivot = df_subset.pivot(index=x_param, columns="section", values="negative_fraction_nz")

            if s1 in pos_pivot.columns and s2 in pos_pivot.columns:
                pos_diff = pos_pivot[s1] - pos_pivot[s2]
                ax.plot(pos_diff.index, pos_diff.values, marker='o',
                        label=f"{dataset} Pos", color=color_dict[dataset])

            if s1 in neg_pivot.columns and s2 in neg_pivot.columns:
                neg_diff = neg_pivot[s1] - neg_pivot[s2]
                ax.plot(neg_diff.index, neg_diff.values, marker='s', linestyle='--',
                        label=f"{dataset} Neg", color=color_dict[dataset])

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(default_params[x_param], color='black', linestyle='--', linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(param_name)
        ax.set_ylabel("Δ (non-zero fractions)")
        #ax.set_ylim(-1, 1)

    # Place the legend higher using bbox_to_anchor
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=len(datasets) * 2, title="Dataset Δ type")

    plt.tight_layout(rect=[0, 0, 1, 1])
    visual.save_or_show_plot(f"output/benchmark_weights_{x_param}.png") #



edge_weight_fractions = pd.read_csv("output/edge_weight_fractions_expr.csv")
defaults = {
    "fpr": 0.02,
    "thres_access": 0.75,
    "degs": 25,
    "max_tf": 50,
    "edge_cutoff_expr": 2000,
    "networktype": "overall"
}

plot_fraction_differences_nz(edge_weight_fractions,
                             x_param="thres_access",
                             param_name="Chromatin Accessibility Threshold",
                             default_params=defaults, width=10)
plot_fraction_differences_nz(edge_weight_fractions, param_name="False Positive Rate",
                             x_param="fpr", default_params=defaults)
plot_fraction_differences_nz(edge_weight_fractions,param_name="DEGs per Comparison",
                             x_param="degs", default_params=defaults)
plot_fraction_differences_nz(edge_weight_fractions,param_name="DATFs per Comparison",
                             x_param="max_tf", default_params=defaults)
plot_fraction_differences_nz(edge_weight_fractions, param_name="Edge Threshold",
                             x_param="edge_cutoff_expr", default_params=defaults)
             