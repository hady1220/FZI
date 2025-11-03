# 🧠 FZI Clustering Suite

A Streamlit-based interactive application for **Flow Zone Indicator (FZI)** analysis and **reservoir flow unit clustering**.  
Designed for **geoscientists, petrophysicists, and reservoir engineers** working with well data (core or logs).

---

## 🚀 Features

✅ **Multi-format Input**
- Accepts: `.csv`, `.xlsx`, `.las`, `.dlis`, `.asc`, `.txt`
- Automatically detects and reads file format
- Preview data and select porosity/permeability columns

✅ **FZI Computation**
- Calculates:
  - **RQI** = 0.0314 × √(K / Φ)
  - **Φ<sub>z</sub>** = Φ / (1 – Φ)
  - **FZI** = RQI / Φ<sub>z</sub>
- Optionally estimates *permeability (K)* from Φ and Vsh when core data are missing

✅ **Multiple Clustering Algorithms**
- **K-Means**
- **Gaussian Mixture Model (GMM)**
- **Agglomerative Hierarchical**
- **HDBSCAN** (if installed)

✅ **Automatic Comparison**
- Runs all clustering methods and compares:
  - Silhouette score
  - Calinski–Harabasz index
  - Davies–Bouldin index
- Selects representative model for each method

✅ **Interactive & Static Visualization**
- **Plotly backend:** Interactive 2×2 crossplots for zooming and hover details  
- **Matplotlib backend:** Publication-ready static figures

✅ **Export**
- Download labeled dataset with assigned cluster numbers
- Summary statistics per cluster (FZI, RQI, Φ<sub>z</sub>)

---

## 🧩 Installation

### 1️⃣ Clone or Download
```bash
git clone https://github.com/<yourusername>/fzi-clustering-suite.git
cd fzi-clustering-suite
