import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import refactored_utils.visualization as visual
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Re-import necessary libraries after code execution state reset
import numpy as np
import matplotlib.pyplot as plt

def analyze_matrix_fractions(matrix: np.ndarray, window: int = 20):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square 2D matrix.")

    n = matrix.shape[0]
    i, j = np.indices((n, n))

    diag_mask = np.abs(i - j) <= window
    upper_mask = (i < j) & (~diag_mask)
    lower_mask = (i > j) & (~diag_mask)

    def get_fractions(values):
        total = len(values)
        non_zero = values[values != 0]
        total_nz = len(non_zero)

        return {
            "zero_fraction": np.mean(values == 0) if total else 0,
            "positive_fraction_nz": np.mean(non_zero > 0) if total_nz else 0,
            "negative_fraction_nz": np.mean(non_zero < 0) if total_nz else 0,
        }

    return {
        "diagonal_window": get_fractions(matrix[diag_mask]),
        "upper_triangle": get_fractions(matrix[upper_mask]),
        "lower_triangle": get_fractions(matrix[lower_mask])
    }
    
def dict_to_df(results):
    records = []

    for name, stat in results.items():
        parts = name.split("_")
        dataset = parts[0]
        networktype = parts[1]
        degs = int(parts[2])

        # thres_access: "05" -> 0.5, "075" -> 0.75, "09" -> 0.9
        thres_access_str = parts[3]
        if len(thres_access_str) == 2:
            thres_access = float(thres_access_str[0] + "." + thres_access_str[1])
        elif len(thres_access_str) == 3:
            thres_access = float(thres_access_str[0] + "." + thres_access_str[1:])
        else:
            raise ValueError(f"Unexpected thres_access format: {thres_access_str}")

        # fpr: "005" -> 0.005, "01" -> 0.01, "02" -> 0.02
        fpr_str = parts[4]
        if len(fpr_str) == 2:
            fpr = float("0." + fpr_str)
        elif len(fpr_str) == 3:
            fpr = float("0.0" + fpr_str[-1])
        elif len(fpr_str) == 4:
            fpr = float("0.00" + fpr_str[-1])
        else:
            raise ValueError(f"Unexpected fpr format: {fpr_str}")

        edge_cutoff_expr = int(parts[5])
        max_tf = int(parts[6])

        for section, values in stat.items():
            records.append({
                "name": name,
                "dataset": dataset,
                "networktype": networktype,
                "degs": degs,
                "thres_access": thres_access,
                "fpr": fpr,
                "edge_cutoff_expr": edge_cutoff_expr,
                "max_tf": max_tf,
                "section": section,
                **values
            })

    return pd.DataFrame(records).sort_values(
        by=["dataset", "networktype", "degs", "thres_access", "fpr", "edge_cutoff_expr", "max_tf", "section"]
    ).reset_index(drop=True)

def sort_matrix_by_gene_rank(matrix_df: pd.DataFrame, gene_rank_df: pd.DataFrame) -> np.ndarray:
    sorted_genes = gene_rank_df.sort_values(by='rank_id')['gene'].tolist()
    pivot_table = matrix_df.pivot(index="Source", columns="Target", values="Coef").fillna(0)
    pivot_table = pivot_table.reindex(index=sorted_genes, columns=sorted_genes, fill_value=0)

    return pivot_table.to_numpy(), sorted_genes


def get_matrix_stats(folder, name, suffix ="exprWeights"):
    df = pd.read_csv(f"{folder}/{name}_edges_final_{suffix}.csv")
    nodes = pd.read_csv(f"{folder}/{name}_nodes_final_{suffix}.csv")

    matrix, sorted_genes = sort_matrix_by_gene_rank(df, nodes)
    return(analyze_matrix_fractions(matrix))



def save_results_file(results, name):                              
    results_df = dict_to_df(results)
    results_df[["dataset", "section", "positive_fraction_nz", "negative_fraction_nz" , "zero_fraction"]]
    results_df.to_csv(name)


folder = "output"
results = {}


results_expr = {}
results_act = {}
for networktype in ["overall", "cellstate"]:
    for dataset in ["Eryth","Mono"]:
        for degs in [50,25,10,5]:
           for thres_access in [0.5,0.75,0.9]:
               for fpr in [0.005, 0.01, 0.02, 0.03]:
                   for edge_cutoff_expr in [2000, 3000, 4000]:
                       for max_tf  in [10,25,50,75,100]: 
                           name = f"{dataset}_{networktype}_{degs}_{thres_access}_{fpr}_{edge_cutoff_expr}_{max_tf}_1000"
                           name = name.replace(".", "")
                           
                           if os.path.isfile(f"{folder}/{name}_edges_final_exprWeights.csv"):
                               results_expr[name] = get_matrix_stats("benchmark_files",name, suffix = "exprWeights")
                               
                           for edge_cutoff_act in [100, 250, 500, 1000]:
                               name = f"{dataset}_{networktype}_{degs}_{thres_access}_{fpr}_{edge_cutoff_expr}_{max_tf}_{edge_cutoff_act}"
                               name = name.replace(".", "")
                               if os.path.isfile(f"results/{name}_edges_final.csv"):
                                   results_act[name] = get_matrix_stats("results",name)


    
save_results_file(results_act, "edge_weight_fractions_act.csv")
save_results_file(results_expr, "edge_weight_fractions_expr.csv")