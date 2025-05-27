import pandas as pd
from utils import get_centralities, multi_compare_plot


gois = ["ZFPM1", "GATA2", "GATA1", "SPI1", "CEBPA", "TAL1", "FLI1",
                    "KLF1", "EGR1", "EGR2", "NAB2","GFI1"]

cols = ["gene", "rank_id", "HSC_TFactivity",
        "Early Erythroid_TFactivity","Late Erythroid_TFactivity"]

def read_file(file,cols):
    df = pd.read_csv(file, index_col=0)
    if cols != None:
        df = df[cols]
    return(df)



edges_overall = read_file("data/Eryth_overall_25_075_002_2000_50_500_edges_final_exprWeights.csv")
                            
edges_cellstate = read_file("data/Eryth_cellstate_25_075_002_2000_50_500_edges_final_exprWeights.csv")

nodes_overall = read_file("data/Eryth_overall_25_075_002_2000_50_500_nodes_final_exprWeights.csv",
                            cols)
nodes_cellstate = read_file("data/Eryth_cellstate_25_075_002_2000_50_500_nodes_final_exprWeights.csv",
                              cols)

nodes_overall = nodes_overall[cols]
nodes_cellstate = nodes_cellstate[cols]

nodes_overall = get_centralities(edges_overall, nodes_overall)
nodes_cellstate = get_centralities(edges_cellstate, nodes_cellstate)

