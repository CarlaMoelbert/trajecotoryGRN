import numpy as np
import pandas as pd

def get_edge_weight(df, prefix='weight'):
    """Calculate average edge weight for columns prefixed with 'weight_' or 'score_'."""
    weight_cols = [col for col in df.columns if col.startswith(f'{prefix}_')]
    if weight_cols:
        df[f'{prefix}_sum'] = df[weight_cols].sum(axis=1)
        df[f'{prefix}'] = df[f'{prefix}_sum'] / len(weight_cols)
        df.drop(columns=[f'{prefix}_sum'], inplace=True)
    return df