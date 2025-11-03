# app.py (updated Streamlit app)
import streamlit as st
import pandas as pd, numpy as np, os
from clustering_suite import read_well_file, compute_FZI, estimate_K, compare_clustering_methods, summarize_clusters

st.set_page_config(page_title="FZI Clustering Suite", layout="wide")
st.title("FZI Clustering Suite (English)")

st.markdown("""
Upload well logs (CSV, Excel, LAS, DLIS, ASCII) or use sample data. Compute FZI and compare clustering methods (KMeans, GMM, Agglomerative, HDBSCAN).
""")

# Sidebar: Data input
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload well file (csv, xlsx, las, dlis, asc, txt)", type=['csv','xlsx','las','dlis','asc','txt'])
use_sample = st.sidebar.button("Load sample data")

# Sidebar: plotting backend
st.sidebar.header("Display options")
plot_backend = st.sidebar.selectbox("Plot backend", options=['plotly','matplotlib'], index=0)

# Parameters
st.sidebar.header("K estimation (if K absent)")
use_k_est = st.sidebar.checkbox("Estimate permeability K from Phi and Vsh", value=False)
C = st.sidebar.number_input("C (empirical)", value=1000.0)
m = st.sidebar.number_input("m", value=4.0)
n = st.sidebar.number_input("n", value=2.0)

st.sidebar.header("Clustering parameters")
ks_text = st.sidebar.text_input("k values (comma) e.g. 2,3,4,5", value="3,4")
hdbscan_min_size = st.sidebar.number_input("HDBSCAN min_cluster_size", value=20, step=1)
run_btn = st.sidebar.button("Run comparison")

# Load data
if use_sample:
    rng = np.random.RandomState(42)
    n = 240
    wells = np.repeat(['W1','W2','W3','W4'], n//4)
    Phi = np.clip(rng.normal(0.18, 0.05, n), 0.01, 0.35)
    K = np.exp(rng.normal(4.5 + (Phi-0.18)*10, 1.0, n))
    Depth = np.tile(np.arange(1000, 1000+n//4), 4)
    Vsh = np.clip(rng.beta(2,5,n), 0, 1)
    df = pd.DataFrame({'Well': wells, 'Depth': Depth, 'Phi': Phi, 'K': K, 'Vsh': Vsh})
    st.success("Sample data loaded.")
    st.dataframe(df.head())
elif uploaded is not None:
    bytes_data = uploaded.read()
    tmp_path = os.path.join('/tmp', uploaded.name)
    with open(tmp_path, 'wb') as f:
        f.write(bytes_data)
    try:
        df = read_well_file(tmp_path)
        st.success(f"Loaded file {uploaded.name}. Preview:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()
else:
    st.info("Upload a well file or click 'Load sample data' to try the app.")

# Proceed if df exists
if 'df' in locals():
    cols = df.columns.tolist()
    st.sidebar.write("Detected columns:", cols)
    phi_col = st.sidebar.selectbox("Porosity column", options=cols, index=cols.index('Phi') if 'Phi' in cols else 0)
    # if K not present user will estimate
    k_col = 'K' if 'K' in df.columns else None

    reservoir_filter = st.sidebar.text_input("Reservoir filter (pandas query) e.g. 'Phi>0.05 and GR<75'", value="")
    if reservoir_filter.strip() != "":
        try:
            df = df.query(reservoir_filter)
            st.sidebar.success("Applied reservoir filter.")
        except Exception as e:
            st.sidebar.error(f"Could not apply filter: {e}")

    # Estimate K if needed
    if use_k_est and 'K' not in df.columns:
        # need Vsh or ask user
        vsh_col = None
        possible = [c for c in cols if c.lower().startswith('vsh') or c.lower().startswith('gr')]
        if possible:
            vsh_col = possible[0]
        else:
            # create default small Vsh
            df['Vsh'] = 0.2
            vsh_col = 'Vsh'
        df['K'] = estimate_K(df[phi_col], df[vsh_col], C=C, m=m, n=n)
        st.sidebar.success("Estimated K from Phi and Vsh.")

    # compute FZI
    try:
        df_f = compute_FZI(df, phi_col=phi_col, k_col='K')
    except Exception as e:
        st.error(f"Error computing FZI: {e}")
        st.stop()

    st.header("FZI basic stats")
    st.write(df_f[['Well','Depth',phi_col,'K','RQI','Phi_z','FZI']].head())

    # Run comparison when requested
    if run_btn:
        ks = [int(x.strip()) for x in ks_text.split(',') if x.strip().isdigit()]
        res = compare_clustering_methods(df_f, ks=ks, hdbscan_min_size=hdbscan_min_size, plot_backend=plot_backend)
        st.success("Clustering comparison done.")

        st.subheader("Representative models and silhouette scores")
        rep = res.get('representative', {})
        rep_table = []
        for method, info in rep.items():
            rep_table.append({'Method': method, 'k': info.get('k'), 'silhouette': float(info.get('score', np.nan)) if info.get('score') is not None else np.nan})
        st.table(pd.DataFrame(rep_table))

        # show figure
        if plot_backend == 'matplotlib' and 'matplotlib_fig' in res:
            st.subheader("Matplotlib comparison plots")
            st.pyplot(res['matplotlib_fig'])
        elif plot_backend == 'plotly' and 'plotly_fig' in res:
            st.subheader("Plotly comparison plots")
            st.plotly_chart(res['plotly_fig'], use_container_width=True)

        # export labeled CSV
        st.subheader("Export labeled data (representative)")
        method_choice = st.selectbox("Choose method to export", options=list(rep.keys()))
        if method_choice:
            labels = res['representative'][method_choice]['labels']
            out_df = df_f.reset_index(drop=True).loc[:len(labels)-1].copy()
            out_df['cluster_method'] = method_choice
            out_df['cluster_label'] = labels
            csv_bytes = out_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download labeled CSV", data=csv_bytes, file_name="fzi_labeled.csv", mime='text/csv')
