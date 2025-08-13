---
title: Data Sources
teaching: 1
exercises: 0
questions:
- "Understanding data sources"
- "How to get data from online sources"
- "How to retrieve dataset with the Toolbox?"
objectives:
- "Brief overview of various data souces"
- "Discuss the benefits and disadvantages of each."
- "Learn to combine Climate data with your own research topic"
- "Learn how to manipulate netCDF data within the CDS Toolbox"
keypoints:
- "Essential libaries for data online data sources"
- "Data retrieval from the CDS Toolbox"
- "netCDF and GRIB data formats"
---

# Materials  Data Sources


## 1) Materials Project 

The Materials Project dataset is a widely used resource in machine learning (ML) for materials science. It provides computed properties and crystal structures for a vast number of materials, derived from quantum mechanical calculations. This dataset currently includes over 70,000 materials with millions of associated properties, making it a rich foundation for developing ML algorithms aimed at materials discovery and prediction.


### Main Features

- **Large and Diverse Coverage** that contains computed properties and crystal structures for over 70,000 inorganic materials, covering diverse chemistries and crystal systems. This diversity allows ML models to capture broad structure-property relationships.

- **High-Quality Computed Properties**  generated from consistent, state-of-the-art quantum mechanical DFT calculations, providing reliable and comparable target properties such as formation energy, band gap, elastic moduli, and phase stability.

- **Uniqueness and Reduced Duplication**  ensures uniqueness of materials and computed properties by reducing duplication, thus enabling models to learn from a comprehensive range of structures without bias toward overrepresented classes.

- **Rich Material Descriptors** that rResearchers often extract domain-specific features or descriptors capturing chemical, structural, and physical characteristics to improve ML model accuracy.

- **Robust Data Access** that the dataset is accessible via a web platform and API with Python tools (e.g., pymatgen), facilitating easy querying, data extraction, and integration into ML workflows.

- **Use in Advanced ML Models** that supports modern techniques including graph neural networks and deep learning on crystal graph representations, which have achieved high accuracy in predicting multiple materials properties.

- **Continuous Growth and Updates** for infrastructure performs thousands of calculations per week, continuously expanding the database, which benefits models trained on increasing data scales and diversity.

- **Reusable for Multiple ML Tasks** beyond basic property prediction, the dataset aids in developing force fields, understanding structure-property relations, and generating new materials candidates via active learning.

  
### Data Source

The Materials Project API (often called the Materials API) is a programmatic interface to access the extensive dataset of materials properties and structures hosted by the Materials Project. It allows users to efficiently query and retrieve various computed materials data without manually using the web interface.

The mp_api is the current official Python client package specifically developed for accessing the Materials Project dataset through their updated RESTful API. Unlike the older pymatgen.ext.matproj module, mp_api is designed to provide a direct and clean interface to the API using modern Python conventions.

Key points about the mp_api usage:

It is packaged as mp_api and can be installed via pip (`pip install mp_api`).

The main client class for interacting with the API is MPRester, imported  to retrieve a material’s structure:

~~~
from mp_api.client import MPRester

with MPRester("your_api_key_here") as mpr:
    structure = mpr.get_structure_by_material_id("mp-1234")
    bandstructure = mpr.get_bandstructure_by_material_id("mp-1234")
~~~
{: .python}

## 2) The Open Molecules 2025 (OMol25)

  - The CPC Global Unified Gauge-Based Analysis of Daily Precipitation is a comprehensive dataset developed by the **Climate Prediction Center (CPC)** of the **National Oceanic and Atmospheric Administration (NOAA)** by taking advantage of the **optimal interpolation (OI)** (> 30,000 gauges (optimal interp. with orographic effects).

  - This dataset provides **global daily** precipitation estimates based on **gauge observations**, offering valuable insights for various applications in **climate research**, **hydrology**, and **weather forecasting**.

   - **Temporal Coverage/Duration**: 1979/01/01 to Present. 

   - **Time Step**: Daily, Monthly

   - **Spatial Resolution**: 0.5x0.5 deg

   - **Missing data**:  flagged with a value of -9.96921e+36f.

   - **Source**: https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html

**Key Strengths**: 

  - High station density

**Key Limitations**: 

  - Quality of the gauge-based analysis is poor over tropical Africa and Antarctica (gauge network density).

### Downloading CPC Data  
 - [Jupyter Note Book Script](https://github.com/mesfind/ml-physical/blob/gh-pages/code/data_source/CPC_global_daily_Precip.ipynb)

---

## 3) OMat24


The Open Materials 2024 (OMat24) dataset contains a mix of single point calculations of non-equilibrium structures and
structural relaxations. The dataset contains structures labeled with total energy (eV), forces (eV/A)
and stress (eV/A^3). The dataset is provided in ASE DB compatible lmdb files.

The OMat24 train and val splits are fully compatible with the Matbench-Discovery benchmark test set.
   1. The splits do not contain any structure that has a protostructure label present in the initial or relaxed
      structures of the WBM dataset.
   2. The splits do not include any structure that was generated starting from an Alexandria relaxed structure with
      protostructure lable in the intitial or relaxed structures of the  WBM datset.

## Subdatasets
OMat24 is made up of X subdatasets based on how the structures were generated. The subdatasets included are:
1. rattled-1000-subsampled & rattled-1000
2. rattled-500-subsampled & rattled-300
3. rattled-300-subsampled & rattled-500
4. aimd-from-PBE-1000-npt
5. aimd-from-PBE-1000-nvt
6. aimd-from-PBE-3000-npt
7. aimd-from-PBE-3000-nvt
8. rattled-relax

**Note** There are two subdatasets for the rattled-< T > datasets. Both subdatasets in each pair were generated with the
same procedure as described in our manuscript.

## File contents and downloads

### OMat24 train split
|       Sub-dataset        | No. structures | File size |                                                                    Download                                                                     |
|:------------------------:|:--------------:|:---------:|:-----------------------------------------------------------------------------------------------------------------------------------------------:|
|      rattled-1000        |    122,937     |   21 GB   |            [rattled-1000.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000.tar.gz)            |
| rattled-1000-subsampled  |     41,786     |  7.1 GB   | [rattled-1000-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000-subsampled.tar.gz) |
|       rattled-500        |     75,167     |   13 GB   |             [rattled-500.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500.tar.gz)             |
|  rattled-500-subsampled  |     43,068     |  7.3 GB   |  [rattled-500-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500-subsampled.tar.gz)  |
|       rattled-300        |     68,593     |   12 GB   |             [rattled-300.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300.tar.gz)             |
|  rattled-300-subsampled  |     37,393     |  6.4 GB   |  [rattled-300-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300-subsampled.tar.gz)  |
|  aimd-from-PBE-1000-npt  |    223,574     |   26 GB   |  [aimd-from-PBE-1000-npt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-npt.tar.gz)  |
|  aimd-from-PBE-1000-nvt  |    215,589     |   24 GB   |  [aimd-from-PBE-1000-nvt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-nvt.tar.gz)  |
|  aimd-from-PBE-3000-npt  |     65,244     |   25 GB   |  [aimd-from-PBE-3000-npt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-npt.tar.gz)  |
|  aimd-from-PBE-3000-nvt  |     84,063     |   32 GB   |  [aimd-from-PBE-3000-nvt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-nvt.tar.gz)  |
|      rattled-relax       |     99,968     |   12 GB   |           [rattled-relax.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-relax.tar.gz)           |
|          Total           |   1,077,382    | 185.8 GB  |

### OMat24 val split (this is a 1M subset used to train eqV2 models from the 5M val split)
**_NOTE:_** The original validation sets contained a duplicated structures. Corrected validation sets were uploaded on 20/12/24. Please see this [issue](https://github.com/facebookresearch/fairchem/issues/942)
for more details, an re-download the correct version of the validation sets if needed.

|       Sub-dataset       |   Size    | File Size |                                                                                                                                      Download |
|:-----------------------:|:---------:|:---------:|----------------------------------------------------------------------------------------------------------------------------------------------:|
|      rattled-1000       |  117,004  |  218 MB   |                       [rattled-1000.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000.tar.gz) |
| rattled-1000-subsampled |  39,785   |   77 MB   | [rattled-1000-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz) |
|       rattled-500       |  71,522   |  135 MB   |                         [rattled-500.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500.tar.gz) |
| rattled-500-subsampled  |  41,021   |   79 MB   |   [rattled-500-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500-subsampled.tar.gz) |
|       rattled-300       |  65,235   |  122 MB   |                         [rattled-300.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300.tar.gz) |
| rattled-300-subsampled  |  35,579   |   69 MB   |   [rattled-300-subsampled.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300-subsampled.tar.gz) |
| aimd-from-PBE-1000-npt  |  212,737  |  261 MB   |   [aimd-from-PBE-1000-npt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-1000-npt.tar.gz) |
| aimd-from-PBE-1000-nvt  |  205,165  |  251 MB   |   [aimd-from-PBE-1000-nvt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-1000-nvt.tar.gz) |
| aimd-from-PBE-3000-npt  |  62,130   |  282 MB   |   [aimd-from-PBE-3000-npt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-npt.tar.gz) |
| aimd-from-PBE-3000-nvt  |  79,977   |  364 MB   |   [aimd-from-PBE-3000-nvt.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-nvt.tar.gz) |
|      rattled-relax      |  95,206   |  118 MB   |                     [rattled-relax.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-relax.tar.gz) |
|          Total          | 1,025,361 |  1.98 GB  |


### sAlex Dataset
We also provide the sAlex dataset used for fine-tuning of our OMat models. sAlex is a subsampled, Matbench-Discovery compliant, version of the original [Alexandria](https://alexandria.icams.rub.de/).
sAlex was created by removing structures matched in WBM and only sampling structure along a trajectory with an energy difference greater than 10 meV/atom. For full details,
please see the manuscript.

| Dataset | Split | No. Structures | File Size |                                                                                               Download |
|:-------:|:-----:|:--------------:|:---------:|-------------------------------------------------------------------------------------------------------:|
|  sAlex  | train |   10,447,765   |  7.6 GB   | [train.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/train.tar.gz) |
|  sAlex  |  val  |    553,218     |  408 MB   |     [val.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/val.tar.gz) |


## Getting ASE atoms objects
Dataset files are written as `AseLMDBDatabase` objects which are an implementation of an [ASE Database](https://wiki.fysik.dtu.dk/ase/ase/db/db.html),
in LMDB format. A single **.aselmdb* file can be read and queried like any other ASE DB.

You can also read many DB files at once and access atoms objects using the `AseDBDataset` class.

For example to read the **rattled-relax** subdataset,
```python
from fairchem.core.datasets import AseDBDataset

dataset_path = "/path/to/omat24/train/rattled-relax"
config_kwargs = {}  # see tutorial on additiona configuration

dataset = AseDBDataset(config=dict(src=dataset_path, **config_kwargs))

# atoms objects can be retrieved by index
atoms = dataset.get_atoms(0)
```

To read more than one subdataset you can simply pass a list of subdataset paths,
```python
from fairchem.core.datasets import AseDBDataset

config_kwargs = {}  # see tutorial on additiona configuration
dataset_paths = [
    "/path/to/omat24/train/rattled-relax",
    "/path/to/omat24/train/rattled-1000-subsampled",
    "/path/to/omat24/train/rattled-1000",
]
dataset = AseDBDataset(config=dict(src=dataset_paths, **config_kwargs))
```
To read all of the OMat24 training or validations splits simply pass the paths to all subdatasets.
