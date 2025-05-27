# trajectoryGRN.py
# Streamlined version of GRN construction pipeline using refactored_utils

import numpy as np
import pandas as pd
import random
import warnings
import os
import argparse
from refactored_utils import *
from trajectoryGRN import GRN
from gimmemotifs.motif import read_motifs
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)



def get_GRN_visuals(nodes, edges,outfolder, check_col= "DATF",
                    name="", gois=None):

    print(f"{nodes.shape} {edges.shape} ")
    get_heatmap(nodes, check_col = check_col, w = 15,
                output=f"{outfolder}/{name}_{check_col}_expression")
    
    if check_col == "DATF":
        print("_________ CHECK COL = DATF _____________")
        print(nodes.columns)
        print(nodes)
        get_heatmap(nodes, suffix ="_TFactivity", check_col = check_col, w= 15,
                            output=f"{outfolder}/{name}_{check_col}_activitys")
        print("________________________________________")
        
    get_adjacency(nodes, edges , w=15, h=15, s=8,
                  output=f"{outfolder}/{name}_Adjacency.png")
    
    get_sankey_plot(edges[edges.Coef > 0], nodes,
                    output=f"{outfolder}/{name}_Sankey_positive.png")
    get_sankey_plot(edges[edges.Coef < 0], nodes,
                    output=f"{outfolder}/{name}_Sankey_negative.png")
    plot_top_centralities(edges, f"{outfolder}/{name}_TopCentralities.png")
    
    if gois != None:
        plot_goi_centralities(edges, gois,f"{outfolder}/{name}_GOICentralities.png")
        get_adjacency(nodes, edges , w=15, h=15, s=8, gois = gois,
                  output=f"{outfolder}/{name}__GOIs_Adjacency.png")


def main():
    # Suppress warnings for cleaner output
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Construct trajectory-aware GRNs.')
    parser.add_argument('--peak2gene_file', type=str, required=True,
                        help='File contraining the relationships between peaks and genes.')
    parser.add_argument('--expr_file', type=str, required=True,
                        help='Gene expression Matrix')
    parser.add_argument('--atac_file', type=str, required=True,
                        help='Peak Accessibility Matrix')
    parser.add_argument('--cellstates', type=str, required=True,
                        help='Comma-separated list of cell states (e.g. "HSC,CMP LMPP,GMP,CD14 Monocyte")')
    parser.add_argument('--eois_file', type=str, default=None,
                        help='')
    parser.add_argument('--networktype', type=str, required=True,
                        help='Network type (e.g. expression, activity)')
    parser.add_argument('--genome', type=str, required=True,
                        help='')
    parser.add_argument('--annotation_column', type=str, required=True,
                        help='')
    parser.add_argument('--fpr', type=float, default=0.02,
                        help='False positive rate for motif filtering')
    parser.add_argument('--thres_access', type=float, default=0.75,
                        help='Accessibility threshold for peak filtering')
    parser.add_argument('--max_features', type=int, default=5,
                        help='Max genes per cellstate')
    parser.add_argument('--max_tf', type=int, default=75,
                        help='Max TFs for final GRN')
    parser.add_argument('--edges_cutoff_expr', type=int, default=4000,
                        help='Cutoff for expression network edges')
    parser.add_argument('--edges_cutoff_act', type=int, default=1000,
                        help='Cutoff for activity network edges')
    parser.add_argument('--getFigures', type=bool, default=False, help='Generate plots')
    parser.add_argument('--get_activity_weights', type=bool, default=False,
                        help='Infer TF activity weights')
    parser.add_argument('--outfolder', type=str, default='figures',
                        help='Output folder for results')
    args = parser.parse_args()

    # Ensure output folder exists
    if not os.path.exists(args.outfolder):
        logging.info(f"Creating new folder at: {args.outfolder}")
        os.makedirs(args.outfolder)

     # In code
    cellstates = args.cellstates.split(",")

    if args.eois_file != None:
        eois = pd.read_csv(args.eois_file, sep=",")
        eois = eois.rename(columns={"Weight":"Direction"})
        gois = set(eois.Source).union(set(eois.Target))
        logging.info(f"Setting the gois to : {args.outfolder}")
        

    
    # Build GRN name
    name = (
        f"{args.networktype}_"
        f"{args.max_features}_"
        f"{args.thres_access}_"
        f"{args.fpr}_"
        f"{args.edges_cutoff_expr}_"
        f"{args.max_tf}_"
        f"{args.edges_cutoff_act}"
    ).replace('.', '')

    # Load motifs and data paths
    motifs = read_motifs()


    # Initialize GRN object
    grn = GRN(
        peak2gene_file=args.peak2gene_file,
        atac_file=args.atac_file,
        expr_file=args.expr_file,
        cellstates=cellstates,
        genome=args.genome,
        motifs=motifs,
        fpr=args.fpr,
        thres_access=args.thres_access,
        layer='normalized',
        annotation_column=args.annotation_column,
        name=name,
        gois=gois,
        edges_cutoff=args.edges_cutoff_expr,
        getFigures=args.getFigures,
        folder=args.outfolder
    )

     # Define the columns
    columns = ["DEGs", "Peaks", "Basenetwork", "Expr_GRN", "TF_TF_expr",
                   "TF_TF_activitiy"]
    if gois != None:
        goi_info = pd.DataFrame(index=list(gois), columns=columns)
        goi_info[:] = None  # Alternatively, you can use goi_info[:] = ""
        grn.goi_info = goi_info
        grn.eois = eois
        
    # Filter and preprocess expression data
    grn.filter_expression(min_genes=10, min_cells=3)
    p2g_og = grn.p2g.copy()

    grn.reduce_to_degs(todense=True, max_features=args.max_features)

    grn.calculate_rank_id()
    grn.update_annotation_with_presence()
    grn.p2g = grn.p2g[grn.p2g.geneName.isin(grn.annotation.gene)]
    
    if gois != None:
        grn.goi_info["DEGs"] = grn.goi_info.index.isin(grn.annotation.gene)

    if (args.getFigures == True)  :
        get_heatmap(grn.annotation, check_col = "DEG", h=3, w=10,
                    output=f"{args.outfolder}/{name}_DEGs_expression.png")
        
    peaks = grn.get_peaks()
    grn.p2g = grn.p2g[(grn.p2g.geneName.isin(grn.expr.var.index)) & (grn.p2g.peakName.isin(grn.atac.var.index))]

    if gois != None:
        grn.goi_info["Peaks"] = grn.goi_info.index.isin(grn.p2g.geneName)

    if (args.getFigures == True)   :
        visualize_peaks(grn.p2g,p2g_og, grn.gois,
                        output=f"{args.outfolder}/{name}_knownPeaks.png")
        
    # Build expression-based GRN
    grn.get_network(grn_type=args.networktype)
    if (args.getFigures == True):
        print(f"Annotation: {grn.annotation.shape}, Edges: {grn.edges_df.shape}")
        print(grn.annotation.DEG.value_counts())
        get_GRN_visuals(nodes= grn.annotation, edges = grn.edges_df,
                outfolder= args.outfolder, name= f"{name}_Expression",
                gois = gois, check_col="DEG")

    # Optionally infer TF activity and build activity-based GRN
    if args.get_activity_weights:
        tfActivity = grn.calculate_tf_activity()
        grn.reduce_to_datfs(max_features=args.max_tf, grn_type='tfs')
        grn.update_annotation_with_presence()
        
        tfrn = predict_links(grn.edges_df,
                             grn.tf_activity,
                             grn.annotation,
                             grn.solver,
                             edges_cutoff=args.edges_cutoff_act)
        tfrn.to_csv(f"{args.outfolder}/{grn.name}_edges_final.csv")
        

        sorted_genes = grn.calculate_rank_id()
        if gois != None:
            grn.goi_info["TF_TF_expr"] = grn.goi_info.index.isin(grn.annotation.gene)
            grn.eois = check_interactions(grn.eois,
                                          grn.edges_df, "TF_TF_expr",
                                          weight_column="Coef")

            
   
        if (args.getFigures == True):
            gene_set = list(set(tfrn.Source).union(set(tfrn.Target)))
            nodes = grn.annotation[grn.annotation.gene.isin(gene_set)]
            get_GRN_visuals(nodes= nodes, edges = tfrn,
                            outfolder= args.outfolder,
                            name= f"{name}_TFactivity", gois = gois)

    # Save expression GRN
    grn.edges_df.to_csv(f"{args.outfolder}/{grn.name}_edges_final_exprWeights.csv")

if __name__ == '__main__':
    main()     