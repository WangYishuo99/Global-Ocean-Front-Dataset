# Global Ocean Front Dataset (2001–2024)

A demonstration repository for the construction of the **Global Daily Ocean Front Dataset (2001–2024)**.

This repository provides a complete example workflow for:

* Multi-algorithm frontal detection
* Probabilistic ensemble construction
* Front-line extraction and vectorization
* Global reconstruction and adaptive seam harmonization

The workflow is demonstrated using a single-day satellite SST sample (`SST_20240101.nc`).

---

## Dataset Description

The full dataset contains:

* Daily front probability fields
* Daily front uncertainty fields
* Vectorized front-line products

covering the global ocean from **2001 to 2024**.

The complete dataset is available from:

**Science Data Bank**

DOI:

`https://doi.org/10.57760/sciencedb.40151`

---

## Repository Structure

```text
.
├── data/
│   ├── SST_20240101.nc
│   ├── ...
│
├── functions/
│   ├── CCA.py
│   ├── CCAIM.py
│   ├── Entropy.py
│   └── ...
│
├── 1_Detection.ipynb
├── 2_Ensemble_Framework.ipynb
├── 3_Frontline_Extraction.ipynb
├── 4_Global_Construction.ipynb
│
├── requirements.txt
│
├── LICENSE
├── CHANGELOG.md
└── README.md
```

---

## Workflow

### 1. Multi-algorithm Frontal Detection

Run:

```bash
1_Detection.ipynb
```

This notebook applies five frontal detection algorithms:

* Cayula-Cornillon Algorithm (CCA)
* CCAIM
* Entropy-based Detection
* Canny Edge Detection
* Gradient-Bayesian-Morphology (GBM)

Outputs:

```text
./data/regions/region_id/algorithm_name.npy
```

---

### 2. Ensemble Framework

Run:

```bash
2_Ensemble_Framework.ipynb
```

This notebook performs:

* Quantile normalization
* Ensemble averaging
* Uncertainty estimation

Outputs:

```text
Ensemble Probability
Ensemble Uncertainty
```

---

### 3. Front-Line Extraction

Run:

```bash
3_Frontline_Extraction.ipynb
```

This notebook demonstrates front-line extraction using the Gulf Stream region (Region 03).

Processing steps include:

* Skeletonization
* Topological reduction
* Front following
* Front merging
* Gap filling
* Ring deletion

Outputs:

```text
Vectorized front lines
GeoJSON examples
```

---

### 4. Global Construction

Run:

```bash
4_Global_Construction.ipynb
```

This notebook demonstrates:

* Global partition merging
* Adaptive seam harmonization
* Probability reconstruction
* Uncertainty reconstruction

Outputs:

```text
Global front probability field
Global front uncertainty field
```

---

## Example Data

The repository includes:

```text
./data/SST_20240101.nc
```

which can be used to reproduce the entire workflow.

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Major packages include:

* numpy
* scipy
* numba
* netCDF4
* rasterio
* geopandas
* shapely
* matplotlib
* scikit-image
* jupyter

See:

```text
requirements.txt
```

for the complete dependency list.

---

## Citation

If you use this repository or the dataset in your research, please cite:

Wang, Y., Zhou, M., & Zhou, F.

*Global Ocean Front Probability, Uncertainty, and Front Line Dataset (2001-2024)*.

Science Data Bank.

DOI: https://doi.org/10.57760/sciencedb.40151

---

## License

This project is released under the MIT License.

See:

```text
LICENSE
```

for details.

---

## Contact

Yishuo Wang

Shanghai Jiao Tong University

Email: [wys1998@sjtu.edu.cn](wys1998@sjtu.edu.cn)
