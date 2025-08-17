---
title: Data Sources
teaching: 1
exercises: 0
questions:
- "Understanding data sources"
- "How to get data from online sources"
- "How to retrieve dataset with the Toolbox?"
objectives:
- "Brief overview of various data sources"
- "Discuss the benefits and disadvantages of each"
- "Learn to combine materials data with your own research"
- "Learn how to manipulate open data through APIs"
keypoints:
- "Essential libraries for accessing online data sources"
- "Data retrieval from the Materials Project API"
- "Working with netCDF, GRIB, and LMDB data formats"
---

# Materials Data Sources

## 1) Materials Project

The Materials Project dataset is a widely used resource in machine learning (ML) for materials science. It provides computed properties and crystal structures for a vast number of materials, derived from quantum mechanical calculations. This dataset currently includes over 70,000 materials with millions of associated properties, making it a rich foundation for developing ML algorithms aimed at materials discovery and prediction.

### Main Features

This dataset comprises computed properties and crystal structures for over 70,000 inorganic materials across diverse chemistries and crystal systems, enabling broad structure-property modeling. All properties derive from consistent, high-quality density functional theory (DFT) calculations, ensuring reliable and comparable targets such as formation energy, band gap, and elastic moduli. The data is curated to minimize duplication, preserving material uniqueness and reducing bias. Researchers can extract rich chemical and structural descriptors to enhance model performance. Accessible via a web platform and API with Python tools (e.g., pymatgen), it supports seamless integration into machine learning workflows. The dataset facilitates advanced techniques, including graph neural networks on crystal graphs, and is continuously updated with thousands of new calculations weekly. Its versatility extends beyond property prediction to force field development, structure-property analysis, and materials discovery via active learning.

### Data Access with MP API

The Materials Project API (often called the Materials API) is a programmatic interface to access the extensive dataset of materials properties and structures hosted by the Materials Project. It allows users to efficiently query and retrieve various computed materials data without manually using the web interface.

The `mp_api` is the current official Python client package specifically developed for accessing the Materials Project dataset through their updated RESTful API. Unlike the older `pymatgen.ext.matproj` module, `mp_api` is designed to provide a direct and clean interface to the API using modern Python conventions. The package is installed via pip with the command:

~~~bash
pip install mp_api
~~~
- It uses the MPRester class to interact with the Materials Project API.

- Access requires logging in at the Materials Project profile site with your email or GitHub account to get your API key.

- The MPRester class is used for all API interactions, preferably within a Python with context manager to properly manage sessions.

- Example code to retrieve a material's structure (e.g., silicon with ID "mp-149"):

~~~
#https://profile.materialsproject.org 
from mp_api.client import MPRester
api_key = "your_api_key_here"
with MPRester(api_key) as mpr:
    structure = mpr.get_structure_by_material_id("mp-1234")
    bandstructure = mpr.get_bandstructure_by_material_id("mp-1234")
~~~
{: .python}


## 3) matminer dataset

The matminer.datasets module is part of the matminer Python library, which is designed for data mining and analysis in the field of materials science. This module provides a growing collection of ready-made materials science datasets. These datasets have been carefully collected and formatted as pandas DataFrames, which are two-dimensional data structures commonly used in data analysis and manipulation.

The key features and benefits of this module include:

- Unified Interface: Users can access many different materials science datasets using a simple and consistent interface.
- Ready-made Datasets: The datasets cover various domains of materials data, such as properties of compounds, crystal structures, and experimental data.
- Pandas DataFrame Format: Each dataset is available as a pandas DataFrame, which means data is organized in tabular form with rows as data points and columns as measured or calculated properties.
- Ease of Use: Loading a dataset is straightforward with a single function call, for example


~~~
from matminer.datasets import load_dataset
df = load_dataset("jarvis_dft_3d")
df.head()
~~~
{: .python}

~~~
   epsilon_x opt  epsilon_y opt  epsilon_z opt                                          structure  ...  epsilon_z tbmbj        mpid gap opt composition
0            NaN            NaN            NaN  [[1.40094192 1.40094192 1.40094192] Co, [0. 0....  ...              NaN  mp-1006883  0.0016    (Co, Ni)
1            NaN            NaN            NaN  [[1.75548056 1.75548056 0.        ] Co, [1.755...  ...              NaN  mp-1008349  0.0018    (Co, Ni)
2            NaN            NaN            NaN  [[0. 0. 0.] Nb, [1.54076297 1.54076297 1.54076...  ...              NaN  mp-1009264  0.0019    (Nb, Co)
3        42.9249        42.9249        42.9249  [[0. 0. 0.] Mg, [1.51711798 1.51711798 1.51711...  ...              NaN  mp-1010953  0.0098    (Mg, Ni)
4        44.0749        44.0749        61.1827  [[0. 0. 0.] Al, [-2.35662665e-06  2.33379406e+...  ...              NaN     mp-1057  0.0143    (Al, Ni)

[5 rows x 16 columns]
~~~
{: .output}

This loads the "jarvis_dft_3d" dataset into a pandas DataFrame named `df`. The datasets module helps researchers and data scientists explore, analyze, and benchmark materials data efficiently without needing to download and preprocess raw data files or interact with database APIs directly. This encourages faster experimentation and model development in materials informatics.

###   Elastic tensor 
~~~
from matminer.datasets.convenience_loaders import load_elastic_tensor
df = load_elastic_tensor()  # loads dataset in a pandas DataFrame object
df.head()
~~~
{: .python}

~~~
  material_id    formula  ...                                     elastic_tensor                            elastic_tensor_original
0    mp-10003    Nb4CoSi  ...  [[311.33514638650246, 144.45092552856926, 126....  [[311.33514638650246, 144.45092552856926, 126....
1    mp-10010  Al(CoSi)2  ...  [[306.93357350984974, 88.02634955100905, 105.6...  [[306.93357350984974, 88.02634955100905, 105.6...
2    mp-10015       SiOs  ...  [[569.5291276937579, 157.8517489654999, 157.85...  [[569.5291276937579, 157.8517489654999, 157.85...
3    mp-10021         Ga  ...  [[69.28798774976904, 34.7875015216915, 37.3877...  [[70.13259066665267, 40.60474945058445, 37.387...
4    mp-10025      SiRu2  ...  [[349.3767766177825, 186.67131003104407, 176.4...  [[407.4791016459293, 176.4759188081947, 213.83...

[5 rows x 17 columns]
~~~
{: .output}


## 4) OMol25 (Open Molecules 2025)

The Open Molecules 2025 (OMol25) dataset contains over 100 million single point calculations of non-equilibrium structures and structural relaxations across a wide swath of organic and inorganic molecular space, including transition metal complexes and electrolytes. The dataset contains structures labeled with total energy (eV) and forces (eV/Å) computed at the wB97M-V/def2-TZVPD level using ORCA6. 

### Key Features
- Massive scale with 100M+ calculations
- Includes non-equilibrium configurations valuable for force field development
- Contains transition metal complexes and electrolytes
- Additional electronic structure data available upon request

### Dataset Format
The dataset is provided in ASE DB compatible LMDB files (*.aselmdb). The dataset contains labels of the total charge and spin multiplicity, saved in the `atoms.info` dictionary because ASE does not support these as default properties.

### Access Information
All information about the dataset is available at the [OMol25 HuggingFace site](https://huggingface.co/facebook/OMol25). If you have issues with the gated model request form, please reach out via a GitHub issue on the repository.

### Calculation Details
To reproduce the calculations:

~~~
from fairchem.data.om.omdata.orca import calc 
~~~
{: .python}

## 3) OMat24 (Open Materials 2024)

The Open Materials 2024 (OMat24) dataset contains a mix of single point calculations of non-equilibrium structures and structural relaxations. The dataset contains structures labeled with total energy (eV), forces (eV/Å) and stress (eV/Å³). 

### Dataset Compatibility
- The train and val splits are fully compatible with the Matbench-Discovery benchmark test set
- Excludes any structure with protostructure labels present in WBM dataset
- Excludes structures generated from Alexandria relaxed structures that appear in WBM

### Subdatasets
OMat24 is composed of several subdatasets based on generation method:

1. **Rattled Structures**:
   - `rattled-1000-subsampled` & `rattled-1000`
   - `rattled-500-subsampled` & `rattled-300`
   - `rattled-300-subsampled` & `rattled-500`
   - `rattled-relax`

2. **AIMD Trajectories**:
   - `aimd-from-PBE-1000-npt`
   - `aimd-from-PBE-1000-nvt`
   - `aimd-from-PBE-3000-npt`
   - `aimd-from-PBE-3000-nvt`


#### sAlex Dataset
A Matbench-Discovery compliant version of Alexandria dataset:

| Dataset | Split | No. Structures | File Size | Download          |
|---------|-------|----------------|-----------|-------------------|
| sAlex   | train | 10,447,765     | 7.6 GB    | [train.tar.gz](...) |
| sAlex   | val   | 553,218        | 408 MB    | [val.tar.gz](...)   |

### Data Access Example

~~~
from fairchem.core.datasets import AseDBDataset

# Load single subdataset
dataset = AseDBDataset(config={
    "src": "/path/to/omat24/train/rattled-relax"
})

# Load multiple subdatasets
dataset = AseDBDataset(config={
    "src": [
        "/path/to/omat24/train/rattled-relax",
        "/path/to/omat24/train/rattled-1000-subsampled"
    ]
})
~~~
{: .python}

## 4) OMC25 (Open Molecular Crystals 2025)

The Open Molecular Crystals 2025 (OMC25) dataset was announced along with UMA, and comprises ~25 million calculations of organic molecular crystals from random packing of OE62 structures into various 3D unit cells. 

### Key Features:
- Calculated at the PBE+D3 level of theory via VASP
- Focuses on organic molecular crystals
- Generated from random packing configurations
- More details and download information coming soon



## Data Preprocessing in Machine Learning

