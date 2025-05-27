import numpy as np
import pandas as pd
import refactored_utils.motif_enrichment as me
from refactored_utils.utils import *
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from refactored_utils.filtering import *

def construct_basenetwork(p2g, motifs, fpr, genome):
    grn, scanned_df = me.motif_enrichment(p2g, motifs, fpr=fpr, ref_genome=genome, divide=100000)
    grn = grn.sort_values(by='score', ascending=False).drop_duplicates(subset=['Source', 'Target'], keep='first')
    grn = grn[grn.Source.isin(p2g.geneName)]
    return grn

def predict_links(grn, expr_matrix, annotation, solver, weight_cutoff=None, edges_cutoff=None):
    TFdict = {key: list(grn.loc[grn.Target == key, 'Source']) for key in set(grn.Target) if key in expr_matrix.columns}
    links_list = []
    for target_gene, reggenes in TFdict.items():
        reggenes = intersect(reggenes, expr_matrix.columns)
        if reggenes:
            X = expr_matrix[reggenes]
            label = expr_matrix[target_gene]
            clf = BaggingRegressor(estimator=Ridge(alpha=1, fit_intercept=True, random_state=42, solver=solver), n_estimators=1000, bootstrap=True, max_features=0.8, n_jobs=-1, random_state=42)
            clf.fit(X, label)
            adj = _get_coef_matrix(clf, reggenes)
            adj = pd.DataFrame(adj, columns=['Coef'])
            adj['Source'] = adj.index
            adj['Target'] = target_gene
            if not adj.empty:
                links_list.append(adj)
    if links_list:
        grn = pd.concat(links_list, ignore_index=True)
    else:
        grn = pd.DataFrame(columns=['Source', 'Coef', 'Target'])
    degs = annotation.loc[annotation.DEG == True, 'gene']
    grn = grn[grn.Target.isin(set(degs).union(set(grn.Source)))]
    grn['Weight'] = grn.Coef.abs()
    if weight_cutoff != None:
        grn = grn[grn.Weight >= weight_cutoff]
    if edges_cutoff != None:
        positive_values = grn[grn['Coef'] > 0].nlargest(edges_cutoff, 'Coef')
        negative_values = grn[grn['Coef'] < 0].nsmallest(edges_cutoff, 'Coef')
        grn = pd.concat([positive_values, negative_values])
        grn = grn.reset_index(drop=True)
    grn = largest_weakly_connected_component(grn)
    grn.rename(columns={'score': 'Score'}, inplace=True)
    grn['Weight'] = grn.Coef.abs()
    grn = largest_weakly_connected_component(grn)
    return grn

def _get_coef_matrix(ensemble_model, feature_names):
    """Helper function to extract coefficients from ensemble model."""
    feature_names = np.array(feature_names)
    n_estimators = len(ensemble_model.estimators_features_)
    coef_list = [pd.Series(ensemble_model.estimators_[i].coef_, index=feature_names[ensemble_model.estimators_features_[i]]) for i in range(n_estimators)]
    coef_df = pd.concat(coef_list, axis=1, sort=False).transpose()
    coef = coef_df.mean()
    return coef