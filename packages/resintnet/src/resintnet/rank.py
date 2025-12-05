
import pandas as pd

def top_mutations(scan_df: pd.DataFrame, k=10):
    return scan_df.nlargest(k, "reward")[["pos","wt","mut","delta_A","stability_penalty","reward"]]

def top_memory_edges(edge_inf_df: pd.DataFrame, k=10):
    return edge_inf_df.nlargest(k, "grad_gmem_A")
