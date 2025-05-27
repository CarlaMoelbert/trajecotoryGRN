import refactored_utils.visualizations as visual
import pandas as pd
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import numpy as np

def merge_dataframes_with_prefix(df1, df2, id_columns, name1, name2):
    # Ask for names for each dataframe

    # Find common columns excluding the id columns
    common_cols = [col for col in df1.columns if col in df2.columns and col not in id_columns]

    # Rename common columns in both dataframes with prefixes
    df1_renamed = df1.rename(columns={col: f"{name1}_{col}" for col in common_cols})
    df2_renamed = df2.rename(columns={col: f"{name2}_{col}" for col in common_cols})

    # Merge dataframes on the id columns
    merged_df = pd.merge(df1_renamed, df2_renamed, on=id_columns, how='inner')
    
    return merged_df


def filter_df(df):
    
    cols = ["dataset", "degs", "thres_access", "fpr", "edge_cutoff_expr", "max_tf",
            "edge_cutoff_act", "edges", "nodes", "genes_found", "edges_found", "edges_correctly_identified", "Modularity", "Network_Density"]
    df = df[df.columns.intersection(cols)]

    return(df)

def compare_networktypes(df, h=3, w=15, output=None,
                         type1 = "Expression", type2="TFActivity",
                         parameters=['dataset', 'fpr', 'thres_access',
                                     'max_features',
                                     'max_tf', "edge_cutoff_expr"],
                         measures=["genes_found",
                                   "edges_found", "edges_correctly_identified",
                                   "Modularity", "Network_Density"]):

    sns.set_style("whitegrid")
    # Ensure measure columns are numeric
    for measure in measures:
        df[f'{type2}_{measure}'] = pd.to_numeric(df[f'{type2}_{measure}'],
                                                    errors='coerce')
        df[f'{type1}_{measure}'] = pd.to_numeric(df[f'{type1}_{measure}'],
                                                    errors='coerce')

    # Generate color palette for datasets (same as `benchmark_parameter`)
    datasets = df["dataset"].unique()
    dataset_palette = dict(zip(datasets, sns.color_palette("tab10", len(datasets))))

    # Create subplots for each measure
    fig, axes = plt.subplots(ncols =len(measures), figsize=(w, h), sharex=False, sharey=False)

    for i, measure in enumerate(measures):
        ax = axes[i]

        # Scatter plot for each dataset
        for dataset, group_df in df.groupby("dataset"):
            ax.scatter(group_df[f'{type1}_{measure}'], group_df[f'{type2}_{measure}'],
                       color=dataset_palette[dataset],alpha=0.1, label=dataset )  # Lower alpha for better visibility

        # Compute valid min/max values for reference line
        overall_vals = df[f'{type1}_{measure}'].dropna()
        cellstate_vals = df[f'{type2}_{measure}'].dropna()

        if not overall_vals.empty and not cellstate_vals.empty:
            min_val = min(overall_vals.min(), cellstate_vals.min())
            max_val = max(overall_vals.max(), cellstate_vals.max())

            ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='grey')

        ax.set_title(measure)
        ax.set_xlabel(type1)
        ax.set_ylabel(type2)

    # Add shared legend for dataset colors
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=dataset_palette[ds], label=ds) for ds in datasets]
    fig.legend(handles=handles, loc="upper center", ncol=len(datasets)//2 + 1, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    visual.save_or_show_plot(output)

def compare_edge_weights(df_expr, df_act, title="", gois=None, output=None, label_type="max_dist", top_n=10):
    merged_df = pd.merge(df_expr, df_act, on=['Source', 'Target'], how='outer')
    merged_df.fillna(0, inplace=True)

    # Load edge orientation info
    eois = pd.read_csv("interactions_formatted_reduced.csv", index_col=0)[["Source", "Target", "Weight"]]
    merged_df = pd.merge(merged_df, eois, on=['Source', 'Target'], how='outer')
    merged_df.fillna(0, inplace=True)

    # Plot setup
    plt.figure(figsize=(4, 4))

  
    grey = merged_df[merged_df['Weight'] == 0]
    plt.scatter(grey['Expr_Weight'], grey['Activity_Weight'], alpha=0.3, color='grey')

    # Plot red and blue points
    red = merged_df[merged_df['Weight'] == 1]
    blue = merged_df[merged_df['Weight'] == -1]
    plt.scatter(red['Expr_Weight'], red['Activity_Weight'],
                alpha=0.6, color='red')
    plt.scatter(blue['Expr_Weight'], blue['Activity_Weight'],
                alpha=0.6, color='blue')

    texts = []

    if label_type == "max_dist":
        merged_df['Diag_Dist'] = np.abs(merged_df['Expr_Weight'] - merged_df['Activity_Weight'])
        top_points = merged_df.nlargest(top_n, 'Diag_Dist')

        for _, row in top_points.iterrows():
            label = f"{row['Source']}-{row['Target']}"
            texts.append(plt.text(row['Expr_Weight'], row['Activity_Weight'], label, fontsize=8, color='black'))

    elif label_type == "max_value":
        top_x = merged_df.nlargest(top_n, 'Expr_Weight')
        top_y = merged_df.nlargest(top_n, 'Activity_Weight')

        top_x_set = set(zip(top_x['Source'], top_x['Target']))
        top_y_set = set(zip(top_y['Source'], top_y['Target']))
        both = top_x_set & top_y_set

        for _, row in merged_df.iterrows():
            key = (row['Source'], row['Target'])
            if key in both:
                color = 'black'
            elif key in top_x_set:
                color = 'green'
            elif key in top_y_set:
                color = 'purple'
            else:
                continue
            label = f"{row['Source']}-{row['Target']}"
            texts.append(plt.text(row['Expr_Weight'], row['Activity_Weight'], label, fontsize=8, color=color))
    elif label_type == "eois":
        
        for _, row in merged_df.iterrows():
            
            if row["Weight"] == 1:
                color = 'red'
            elif row["Weight"] == -1:
                color = "blue"
            else:
                continue
            label = f"{row['Source']}-{row['Target']}"
            texts.append(plt.text(row['Expr_Weight'], row['Activity_Weight'],
                                  label, fontsize=8, color=color))

    min_val = min(merged_df['Expr_Weight'].min(), merged_df['Activity_Weight'].min())
    max_val = max(merged_df['Expr_Weight'].max(), merged_df['Activity_Weight'].max())

    # Adjust text to avoid overlaps
    adjust_text(
        texts,
        expand_points=(1.2, 1.4),
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
    )

    # Set labels and title
    plt.xlabel('Expression-based Weight', fontsize=8)
    plt.ylabel('TF Activity-based Weight', fontsize=8)
    plt.tick_params(axis='both', labelsize=8)
    plt.title(title, fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    visual.save_or_show_plot(output)
