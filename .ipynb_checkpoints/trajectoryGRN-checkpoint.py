# Standard library
import logging
import random
import warnings
from functools import reduce
from itertools import product

# Scientific computing
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize, scale

# External tools
import decoupler as dc


# Custom utilities
from refactored_utils import *


warnings.simplefilter(action="ignore", category=FutureWarning)

np.random.seed(42)  # Set a seed for numpy
random.seed(42)     # Set a seed for python's random module
from matplotlib.colors import TwoSlopeNorm

class GRN:
    def __init__(self, 
                 peak2gene_file,
                 atac_file,
                 expr_file,
                 cellstates,
                 genome, 
                 motifs, 
                 annotation_column="annotation",
                 meta=None, 
                 thres_access=0.9,
                 thres_score=10,
                 fpr=0.01,
                 weight_cutoff = None,
                 solver="sparse_cg",
                 layer="scaled",
                 filtering = False,
                 edges_cutoff = None,
                 log_level=logging.INFO,
                 gois =["GATA1", "GATA2", "SPI1"],
                 folder = "figures",
                 name = "GRN",
                 coexpr = 0,
                 getFigures = False):
        
        # Initialize logging
        self.log_level = log_level
        logging.basicConfig(level=self.log_level)
        logging.info(f"Initializing GRNSummary for cell states: {cellstates}")

        # Assign parameters
        self.cellstates = cellstates
        self.name = name
        self.genome, self.motifs, self.annotation_column = genome, motifs, annotation_column
        self.p2g = get_peak2gene(peak2gene_file)
        
        logging.info(f"GET ATAC data")
        self.atac = get_atac(atac_file,
                                   self.p2g, cellstates,
                                   annotation_column = self.annotation_column)

        logging.info(f"Get Expr data")
        self.expr = get_expression(expr_file,
                                   meta=meta, celltypes=cellstates,
                                   filtering = False,
                                   annotation_column = self.annotation_column)

        logging.info(f"Make sure names are unique")
        if not self.atac.var_names.is_unique:
            self.atac = self.atac[:, ~self.atac.var_names.duplicated()].copy()
        if not self.expr.var_names.is_unique:
            self.expr = self.expr[:, ~self.expr.var_names.duplicated()].copy()

        logging.info(f"Set Parameters")
        self.p2g = self.p2g[(self.p2g.geneName.isin(self.expr.var.index)) & (self.p2g.peakName.isin(self.atac.var.index)) ]


        self.thres_access = thres_access
        self.fpr = fpr
        self.solver = solver
        self.layer = layer
        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self.gois = gois
        self.weight_cutoff = weight_cutoff
        self.edge_cutoff = edges_cutoff
        self.getFigures = getFigures
        logging.info("Setting up annotation")
        self.set_annotation()
        logging.info("Done setting annotation")
        self.coexpr = coexpr
        self.folder = folder

        self.node_colors = {}
        self.edge_colors = {}
        logging.info(f"Done with initialisation")


    def set_annotation(self):
        self.annotation = add_TF_information(
            self.expr.var[["gene"]].copy()[self.expr.var["gene"].isin(self.p2g.geneName)], 
            self.motifs
        )
    def filter_expression(self, min_genes, min_cells, filter_mito=True, mito_threshold=10):
        logging.info(f"Filter expression...")
        self.expr = filter_cells(self.expr, min_genes, min_cells)
        logging.info(f"Filter expression...")
        self.p2g = self.p2g[(self.p2g.geneName.isin(self.expr.var.index))]
        logging.info(f"Filter expression... ")
    
    def filter_for_genes(self, genes):
        logging.info(f"Filter for genes...")
        # Ensure unique var names in ATAC
        if not self.atac.var_names.is_unique:
            self.atac = self.atac[:, ~self.atac.var_names.duplicated()].copy()
    
        peak_names = self.p2g.loc[self.p2g.geneName.isin(genes), "peakName"]
        
        # Ensure peak_names are unique
        peak_names = peak_names.drop_duplicates()
        peak_names = peak_names[peak_names.isin(self.atac.var.index)]
        
        # Subset the ATAC data using unique peak names
        self.atac = self.atac[:,peak_names].copy() 
        
        # Filter the annotation DataFrame"
        self.annotation = self.annotation[self.annotation["gene"].isin(genes)]

    def reduce_to_degs(self, todense=False, max_features = 10):
        logging.info(f"Reduce to DEGs....")
        self.expr.X = self.expr.layers["raw_count"]
        
        degs = self.select_differential_features(self.expr.to_df(), "DEG",
                                                 threshold=0.05,
                                                 max_features=max_features)
        genes = degs.union(self.annotation.loc[self.annotation.TF == True, "gene"])
        self.filter_for_genes(genes)
        
        self.expr.layers["normalized"] = normalize(self.expr.layers["raw_count"],
                                                   axis=0, norm='l2') 
        
        if sp.issparse(self.expr.layers["normalized"]):
            self.expr.layers["normalized"] = self.expr.layers["normalized"].toarray()
        self.expr.layers["scaled"] = scale(self.expr.layers["normalized"], axis=0) 

    def reduce_to_datfs(self, todense=False, max_features=10, grn_type="all"):
        logging.info(f"Reduce to DATFs...")
        degs = self.select_differential_features(self.tf_activity, "DATF", 
                                                 threshold=0.05,
                                                 max_features=max_features)
        if grn_type == "all":
            filtered_genes = self.annotation[(self.annotation["DEG"]) | (self.annotation["DATF"])]["gene"]
        elif grn_type == "tfs":
            filtered_genes = self.annotation[(self.annotation["DATF"])]["gene"]

        self.filter_for_genes(filtered_genes)
        self.edges_df = self.edges_df[(self.edges_df.Source.isin(filtered_genes)) & (self.edges_df.Target.isin(filtered_genes))]

    def update_annotation_with_presence(self):
        logging.info(f"Update Annotation with presence ...")
        cell_info = self.expr.obs[self.annotation_column]
        for cell_state in self.cellstates:
            cells_in_state = cell_info[cell_info == cell_state].index
            mean_expression = self.expr[cells_in_state, :].layers[self.layer].mean(axis=0)
            mean_expression = mean_expression.A1 if sp.issparse(mean_expression) else mean_expression.flatten()
            expression_series = pd.Series(mean_expression, index=self.expr.var.index)
            self.annotation[f"{cell_state}_expression"] = self.annotation["gene"].map(expression_series)

            if hasattr(self, 'tf_activity'):
                mean_tf_activity = self.tf_activity.loc[cells_in_state].mean(axis=0)
                tf_activity_series = pd.Series(mean_tf_activity, index=self.tf_activity.columns)
                self.annotation[f"{cell_state}_TFactivity"] = self.annotation["gene"].map(tf_activity_series)
    
    def get_peaks(self):
        logging.info(f"Get Peaks...")
        tfs = set(self.annotation.loc[self.annotation.TF == True, "gene"])
        degs =  set(self.annotation.loc[self.annotation.DEG == True, "gene"])

        
        p2g = self.p2g.copy()
        genes = set(degs).union(set(tfs))#
        p2g = p2g[p2g.geneName.isin(genes)]
        p2gs = {}
        peaks = {}
        
        for state in set(self.expr.obs[self.annotation_column]):
            p2gs[state] = get_peak_subset(self.atac, p2g, self.annotation_column,
                                          state, self.thres_access)
            peaks[state] = set(p2gs[state].peakName)

        
        p2g_sub = reduce(lambda left,right: pd.merge(left,right,how='outer', sort=True),
                         p2gs.values())

        p2g_sub = p2g_sub.fillna(False)
        self.atac = self.atac[:, list(set(p2g_sub.peakName))]
        self.expr = self.expr[:, list(set(p2g_sub.geneName))]

        self.p2g = p2g_sub
        self.cellstate_p2g = p2gs
        
        if self.getFigures == True:
            output_filename = f"{self.folder}/{self.name}_peaks.png"
            show_peak_upset(peaks, output=output_filename)
        return(peaks)

    def get_subnetwork(self, p2g, prefix = None):
        grn = construct_basenetwork(p2g, self.motifs, self.fpr, self.genome)
        grn, expr_matrix = filter_basenetwork(grn, self.expr, self.annotation,
                                              self.layer, gois=self.gois,
                                              filter_coexpr =self.coexpr)

        if hasattr(self, 'goi_info') and self.goi_info is not None:
            goi_name = "Basenetwork"
            if prefix != None:
                goi_name = f"{prefix}_{goi_name}"
                
            self.goi_info[goi_name] = self.goi_info.index.isin(set(grn.Source).union(set(grn.Target)))
            self.eois = check_interactions(self.eois, grn, goi_name,
                                           weight_column=None)
            
        if (self.gois != None) & (self.getFigures== True): #( 
            get_adjacency(self.annotation, grn, gois = self.gois,
                              w=6, h=6,
                              output=f"{self.folder}/{self.name}_basenetwork_full.png")
            self.annotation.to_csv(f"{self.folder}/{self.name}_basenetwork_nodes.csv")
            grn.to_csv(f"{self.folder}/{self.name}_basenetwork.csv")

        logging.info(f"Predict links...")
        grn = predict_links(grn, expr_matrix, self.annotation, self.solver,
                            weight_cutoff = self.weight_cutoff,
                           edges_cutoff = self.edge_cutoff)
        
        if hasattr(self, 'goi_info') and self.goi_info is not None:
            goi_name = "Expr_GRN"
            if prefix != None:
                goi_name = f"{prefix}_{goi_name}"
                
            self.goi_info[goi_name] = self.goi_info.index.isin(set(grn.Source).union(set(grn.Target)))
            self.eois = check_interactions(self.eois, grn, goi_name, weight_column="Coef")
            
        if self.getFigures == True:
            show_distribution(grn, "Source", f"{self.folder}/{self.name}", "grn")
            show_distribution(grn, "Target", f"{self.folder}/{self.name}", "grn")
            show_distribution(grn, "Coef", f"{self.folder}/{self.name}", "grn",
                                  counts = False)

        self.annotation.to_csv(f"{self.folder}/{self.name}_withNonTF_nodes.csv")
        grn.to_csv(f"{self.folder}/{self.name}_withNonTF_edges.csv")
        return grn
        
    def get_network(self, grn_type="overall"):
        
        if grn_type == "overall":
            logging.info(f"Get overall network...")
            grn = self.get_subnetwork(self.p2g)
            
        elif grn_type == "cellstate":
            grns = {}
            for state in self.cellstates:
                logging.info(f"Get network of state {state}")
                grns[state] = self.get_subnetwork(self.cellstate_p2g[state], prefix=state)
            grn = combine_all(grns)
            
            if hasattr(self, 'goi_info') and self.goi_info is not None:
                basenetwork_cols = [col for col in self.goi_info.columns if col.endswith("_Basenetwork")]
                self.goi_info["Basenetwork"] = self.goi_info[basenetwork_cols].any(axis=1)
                
                basenetwork_cols = [col for col in self.goi_info.columns if col.endswith("_Expr_GRN")]
                self.goi_info["Expr_GRN"] = self.goi_info[basenetwork_cols].any(axis=1)

                
        self.edges_df = grn
        genes = set(grn.Source).union(set(grn.Target))
        self.expr = self.expr[:, list(genes)]
      
    
    def select_differential_features(self, data_matrix, annotation_col_name,
                                     threshold=0.05, max_features=50):
        logging.info(f"Select differential features...")
        cell_info = self.expr.obs[self.annotation_column].copy()

        
        degs = set()
        degs_lists = {}
        summary = {}
        for i, cell_state in enumerate(self.cellstates):
            cells_in_state = cell_info[cell_info == cell_state].index
            position = i + 1

            
                
            if i > 0:
                other_states = self.cellstates[i -1]
                features_right = get_degs(data_matrix, cells_in_state,
                                          cell_info[cell_info ==(other_states)].index,
                                          threshold, max_features,
                                          name=f"{cell_state}_{annotation_col_name}_right",
                                          output=self.name, gois = self.gois,
                                          saveFigures = self.getFigures,
                                          folder=self.folder)
                
                self.update_annotation(features_right, f"{cell_state}_right",
                                       cell_state, position + 0.25)
            
                degs = degs.union(features_right)
                degs_lists[f"{cell_state}_right"] = features_right 
                
            if i < len(self.cellstates) - 1:
                other_states = self.cellstates[i + 1]
                features_left = get_degs(data_matrix, cells_in_state,
                                         cell_info[cell_info == (other_states)].index,
                                         threshold, max_features,
                                         name=f"{cell_state}_{annotation_col_name}_left",
                                         output=self.name, 
                                         gois = self.gois,
                                         saveFigures = self.getFigures,folder=self.folder)
          
                self.update_annotation(features_left, f"{cell_state}_left",
                                       cell_state, position - 0.25)
                
                degs = degs.union(features_left)
                degs_lists[f"{cell_state}_left"] = features_left
                
        if self.getFigures== True:
            show_peak_upset(degs_lists,
                                output=f"{self.folder}/{self.name}_upset_{annotation_col_name}.png")
        
        self.annotation[annotation_col_name] = self.annotation["gene"].isin(degs).copy()
        return degs

    def update_annotation(self, selected_features, col_name, cell_state, position):
        logging.info(f"Update annotation...")
        annotation = self.annotation.copy()
        if col_name not in annotation:
            annotation[col_name] = 0.0
        annotation.loc[annotation["gene"].isin(selected_features), col_name] = float(position)
        self.annotation = annotation

    def calculate_rank_id(self):
        logging.info(f"Calculate Rank id...")
        rank_ids = pd.Series(0, index=self.annotation["gene"])
        non_zero_counts = pd.Series(0, index=self.annotation["gene"])
    
        for cell_state in self.cellstates:
            right_col = f"{cell_state}_right"
            left_col = f"{cell_state}_left"
            
            # Check if columns exist in annotation and count non-zero occurrences
            if right_col in self.annotation.columns:
                right_data = self.annotation[right_col].fillna(0)
                rank_ids += right_data
                non_zero_counts += (right_data != 0).astype(int)
                
            if left_col in self.annotation.columns:
                left_data = self.annotation[left_col].fillna(0)
                rank_ids += left_data
                non_zero_counts += (left_data != 0).astype(int)
    
        # Avoid division by zero by replacing zeros in non_zero_counts with 1 temporarily
        non_zero_counts = non_zero_counts.replace(0, 1)
        rank_ids = rank_ids / non_zero_counts
        
        self.annotation["rank_id"] = rank_ids
        self.annotation = self.annotation.sort_values("rank_id")
    
        # Sort genes by rank_id
        sorted_genes = self.annotation.sort_values("rank_id")["gene"]
        return sorted_genes    
   
    def calculate_tf_activity(self, todense=True):
        logging.info(f"Calculate TF activity....")
        genes_in_grn = set(self.edges_df["Source"]).union(self.edges_df["Target"])

        
        exp = get_expr_matrix(self.expr[:, list(genes_in_grn)], todense=False,
                              layer="scaled")
        index = exp.index
        col = exp.columns
        exp = normalize(exp, axis= 0, norm='l2')
        exp = scale(pd.DataFrame(exp), axis=0)
        exp = pd.DataFrame(exp, index=index, columns=col)
        random.seed(42)
        TF_activity = dc.run_ulm(mat = exp, net = self.edges_df, min_n=1,
                                 source='Source',
                                 target='Target',
                                 weight='Coef',
                                 verbose=False,
                                 use_raw=False)
        activity_TF = TF_activity[0] #exp.obsm['wmean_estimate'].copy()
    
        TF_activity = activity_TF.copy()
        TF_activity[TF_activity.isnull()] = 0
        TF_activity = pd.DataFrame(TF_activity, index=activity_TF.index,
                                   columns=activity_TF.columns)
        self.tf_activity = TF_activity
        return(TF_activity)