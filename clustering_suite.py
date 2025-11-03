# clustering_suite.py (updated)
# - file readers for CSV/Excel/LAS/DLIS/ASCII
# - compute FZI / RQI / Phi_z
# - clustering methods: KMeans, GMM, Agglomerative, HDBSCAN (if available)
# - compare_clustering_methods: returns Matplotlib figure and Plotly figure for side-by-side comparison

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Optional libs
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

# -------------------------
# File readers
# -------------------------
def read_csv_excel(path):
    path = str(path)
    if path.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(path)
    else:
        return pd.read_csv(path)

def read_las(path):
    try:
        import lasio
    except Exception as e:
        raise ImportError("lasio is required to read LAS files. Install with `pip install lasio`.") from e
    las = lasio.read(path)
    # las.df() returns curves; ensure we have a regular DataFrame
    try:
        df = las.df()
    except Exception:
        df = las.df()
    # some LAS have depth index; reset to columns
    df = df.reset_index().rename(columns={'DEPT':'Depth','DEPTH':'Depth'}) if ('DEPT' in df.index.names or 'DEPTH' in df.index.names) else df.reset_index()
    return df

def read_dlis(path):
    try:
        import dlisio
    except Exception as e:
        raise ImportError("dlisio is required to read DLIS files. Install with `pip install dlisio`.") from e
    ds = dlisio.DLISFile(path)
    all_series = {}
    try:
        for logical in ds.logical_records:
            for curve in logical.curves:
                name = getattr(curve, 'mnemonic', None) or getattr(curve, 'name', None) or 'curve'
                vals = getattr(curve, 'values', None)
                if vals is not None:
                    all_series[name] = vals
    except Exception:
        pass
    if not all_series:
        raise ValueError("Could not extract curves from DLIS file")
    df = pd.DataFrame(all_series)
    return df

def read_ascii(path):
    # try common delimiters
    try:
        df = pd.read_csv(path, sep=None, engine='python', comment='#')
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, comment='#', header=0)
    return df

def read_well_file(path):
    """
    Detect and read a well file. Supported extensions:
    .csv, .xlsx, .las, .dlis, .asc, .txt
    """
    path = str(path)
    lower = path.lower()
    if lower.endswith(('.csv', '.txt')):
        return read_csv_excel(path)
    if lower.endswith(('.xls', '.xlsx')):
        return read_csv_excel(path)
    if lower.endswith('.las'):
        return read_las(path)
    if lower.endswith('.dlis'):
        return read_dlis(path)
    if lower.endswith(('.asc', '.txt')):
        return read_ascii(path)
    raise ValueError("Unsupported file extension. Supported: CSV, XLSX, LAS, DLIS, ASC, TXT")

# -------------------------
# Core computations
# -------------------------
def estimate_K(Phi, Vsh, C=1000, m=4, n=2):
    """
    Empirical K estimation from Phi & Vsh:
    K = C * Phi^m * (1 - Vsh)^n
    """
    return C * (Phi ** m) * ((1 - Vsh) ** n)

def compute_FZI(df, phi_col='Phi', k_col='K'):
    """
    Compute RQI, Phi_z, FZI and log versions.
    Expects numeric Phi and K columns (Phi as fraction 0-1).
    """
    df = df.copy()
    if phi_col not in df.columns or k_col not in df.columns:
        raise KeyError(f"Required columns not found: {phi_col}, {k_col}")
    df[phi_col] = pd.to_numeric(df[phi_col], errors='coerce')
    df[k_col] = pd.to_numeric(df[k_col], errors='coerce')
    df = df.dropna(subset=[phi_col, k_col]).copy()
    df[phi_col] = df[phi_col].clip(lower=1e-6, upper=0.999999)
    df[k_col] = df[k_col].clip(lower=1e-12)
    df['RQI'] = 0.0314 * np.sqrt(df[k_col] / df[phi_col])
    df['Phi_z'] = df[phi_col] / (1 - df[phi_col])
    df['FZI'] = df['RQI'] / df['Phi_z']
    df['logRQI'] = np.log10(df['RQI'])
    df['logPhi_z'] = np.log10(df['Phi_z'])
    df['logFZI'] = np.log10(df['FZI'])
    return df

# -------------------------
# Clustering helpers
# -------------------------
def _prepare_X(df):
    # use logRQI & logPhi_z
    X = df[['logRQI','logPhi_z']].dropna().values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def run_kmeans(df, ks=[2,3,4], random_state=42):
    Xs, _ = _prepare_X(df)
    results = {}
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state).fit(Xs)
        labels = km.labels_
        results[k] = {'labels': labels, 'model': km,
                      'silhouette': silhouette_score(Xs, labels) if k>1 else np.nan,
                      'ch': calinski_harabasz_score(Xs, labels),
                      'db': davies_bouldin_score(Xs, labels)}
    return results

def run_gmm(df, ks=[2,3,4], random_state=42):
    Xs, _ = _prepare_X(df)
    results = {}
    for k in ks:
        g = GaussianMixture(n_components=k, random_state=random_state).fit(Xs)
        labels = g.predict(Xs)
        results[k] = {'labels': labels, 'model': g,
                      'bic': g.bic(Xs), 'aic': g.aic(Xs),
                      'silhouette': silhouette_score(Xs, labels) if k>1 else np.nan,
                      'ch': calinski_harabasz_score(Xs, labels),
                      'db': davies_bouldin_score(Xs, labels)}
    return results

def run_agglomerative(df, ks=[2,3,4]):
    Xs, _ = _prepare_X(df)
    results = {}
    for k in ks:
        agg = AgglomerativeClustering(n_clusters=k).fit(Xs)
        labels = agg.labels_
        results[k] = {'labels': labels, 'model': agg,
                      'silhouette': silhouette_score(Xs, labels) if k>1 else np.nan,
                      'ch': calinski_harabasz_score(Xs, labels),
                      'db': davies_bouldin_score(Xs, labels)}
    return results

def run_hdbscan(df, min_cluster_size=20, min_samples=None):
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan not available. Install `pip install hdbscan`.")
    Xs, _ = _prepare_X(df)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(Xs)
    res = {'labels': labels, 'model': clusterer}
    unique = set(labels)
    if -1 in unique:
        unique.remove(-1)
    if len(unique) > 1:
        lab_f = labels[labels>=0]
        Xs_f = Xs[labels>=0]
        res.update({'silhouette': silhouette_score(Xs_f, lab_f),
                    'ch': calinski_harabasz_score(Xs_f, lab_f),
                    'db': davies_bouldin_score(Xs_f, lab_f)})
    else:
        res.update({'silhouette': np.nan, 'ch': np.nan, 'db': np.nan})
    return res

# -------------------------
# Comparison & plotting
# -------------------------
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.express as px

def _attach_labels_to_df(df, labels):
    df2 = df[['RQI','Phi_z','FZI']].dropna().reset_index(drop=True).copy()
    df2['Cluster'] = labels
    return df2

def compare_clustering_methods(df, ks=[3,4], hdbscan_min_size=20, plot_backend='plotly'):
    '''
    Run KMeans, GMM, Agglomerative (for each k in ks) and HDBSCAN (if available).
    Returns a dict of results and either a Plotly figure or Matplotlib figure.
    plot_backend: 'plotly' or 'matplotlib'
    '''
    results = {}
    km_res = run_kmeans(df, ks=ks)
    gm_res = run_gmm(df, ks=ks)
    ag_res = run_agglomerative(df, ks=ks)
    results['KMeans'] = km_res
    results['GMM'] = gm_res
    results['Agglomerative'] = ag_res
    if HDBSCAN_AVAILABLE:
        try:
            results['HDBSCAN'] = run_hdbscan(df, min_cluster_size=hdbscan_min_size)
        except Exception as e:
            results['HDBSCAN_error'] = str(e)

    # select representative label sets: choose k with highest silhouette for each method (if available)
    representative = {}
    for method, res in [('KMeans',km_res), ('GMM',gm_res), ('Agglomerative',ag_res)]:
        best_k = None; best_score = -999
        for k, rr in res.items():
            sc = rr.get('silhouette', -999)
            if sc is None: sc = -999
            if sc > best_score:
                best_score = sc; best_k = k
        representative[method] = {'k': best_k, 'labels': res[best_k]['labels'], 'score': best_score}

    if 'HDBSCAN' in results and isinstance(results['HDBSCAN'], dict):
        representative['HDBSCAN'] = {'k': len(set(results['HDBSCAN']['labels'])),
                                    'labels': results['HDBSCAN']['labels'],
                                    'score': results['HDBSCAN'].get('silhouette', np.nan)}

    # prepare DataFrames for plotting
    dfs = {}
    for method, info in representative.items():
        dfs[method] = _attach_labels_to_df(df, info['labels'])

    # plotting
    methods = list(dfs.keys())[:4]
    if plot_backend == 'matplotlib':
        fig, axes = plt.subplots(2,2, figsize=(12,10))
        axes = axes.flatten()
        for i, method in enumerate(methods):
            dfi = dfs[method]
            axes[i].scatter(np.log10(dfi['RQI']), np.log10(dfi['Phi_z']), c=dfi['Cluster'], cmap='tab10', s=15)
            axes[i].set_title(f"{method} (k={representative[method]['k']})  sil={representative[method]['score']:.3f}")
            axes[i].set_xlabel('log10(RQI)'); axes[i].set_ylabel('log10(Phi_z)')
            axes[i].grid(alpha=0.3)
        plt.tight_layout()
        return {'results': results, 'representative': representative, 'matplotlib_fig': fig}
    else:
        # plotly subplot 2x2
        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=[f"{m} (k={representative[m]['k']}) sil={representative[m]['score']:.3f}" for m in methods])
        pos = [(1,1),(1,2),(2,1),(2,2)]
        for i, method in enumerate(methods):
            dfi = dfs[method]
            row, col = pos[i]
            scatter = px.scatter(dfi, x=np.log10(dfi['RQI']), y=np.log10(dfi['Phi_z']), color=dfi['Cluster'].astype(str),
                                 labels={'x':'log10(RQI)','y':'log10(Phi_z)'})
            for trace in scatter.data:
                fig.add_trace(trace, row=row, col=col)
        fig.update_layout(height=800, width=1000, showlegend=False, title_text="Clustering comparison")
        return {'results': results, 'representative': representative, 'plotly_fig': fig}

def summarize_clusters(df):
    if 'Cluster' not in df.columns:
        return pd.DataFrame()
    summary = df.groupby('Cluster')[['FZI','RQI','Phi_z']].agg(['count','mean','std']).reset_index()
    return summary
