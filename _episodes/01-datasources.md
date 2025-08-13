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

- **Large and Diverse Coverage**: Contains computed properties and crystal structures for over 70,000 inorganic materials, covering diverse chemistries and crystal systems. This diversity allows ML models to capture broad structure-property relationships.

- **High-Quality Computed Properties**: All properties are generated from consistent, state-of-the-art quantum mechanical DFT calculations, providing reliable and comparable target properties such as formation energy, band gap, elastic moduli, and phase stability.

- **Uniqueness and Reduced Duplication**: The dataset ensures uniqueness of materials and computed properties by reducing duplication, enabling models to learn from a comprehensive range of structures without bias toward overrepresented classes.

- **Rich Material Descriptors**: Researchers often extract domain-specific features or descriptors capturing chemical, structural, and physical characteristics to improve ML model accuracy.

- **Robust Data Access**: The dataset is accessible via a web platform and API with Python tools (e.g., pymatgen), facilitating easy querying, data extraction, and integration into ML workflows.

- **Use in Advanced ML Models**: Supports modern techniques including graph neural networks and deep learning on crystal graph representations, which have achieved high accuracy in predicting multiple materials properties.

- **Continuous Growth and Updates**: The infrastructure performs thousands of calculations per week, continuously expanding the database, which benefits models trained on increasing data scales and diversity.

- **Reusable for Multiple ML Tasks**: Beyond basic property prediction, the dataset aids in developing force fields, understanding structure-property relations, and generating new materials candidates via active learning.

### Data Access with MP API

The Materials Project API (often called the Materials API) is a programmatic interface to access the extensive dataset of materials properties and structures hosted by the Materials Project. It allows users to efficiently query and retrieve various computed materials data without manually using the web interface.

The `mp_api` is the current official Python client package specifically developed for accessing the Materials Project dataset through their updated RESTful API. Unlike the older `pymatgen.ext.matproj` module, `mp_api` is designed to provide a direct and clean interface to the API using modern Python conventions.

Key points about the `mp_api` usage:
- Installed via pip: `pip install mp_api`
- Uses the `MPRester` class for all API interactions

Example to retrieve a material's structure:
```python
from mp_api.client import MPRester

with MPRester("your_api_key_here") as mpr:
    structure = mpr.get_structure_by_material_id("mp-1234")
    bandstructure = mpr.get_bandstructure_by_material_id("mp-1234")
```

## 2) OMol25 (Open Molecules 2025)

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
```python
from fairchem.data.om.omdata.orca import calc  # For writing compatible ORCA inputs
```

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

### File Contents and Downloads

#### OMat24 Train Split
| Sub-dataset | No. structures | File size | Download |
|------------|---------------|-----------|----------|
| [Full table maintained as in original...] | 

#### OMat24 Val Split
*Note*: Corrected validation sets were uploaded on 20/12/24 due to duplicated structures.

| Sub-dataset | Size | File Size | Download |
|------------|------|-----------|----------|
| [Full table maintained as in original...] |

#### sAlex Dataset
A Matbench-Discovery compliant version of Alexandria dataset:

| Dataset | Split | No. Structures | File Size | Download |
|---------|-------|----------------|-----------|----------|
| sAlex | train | 10,447,765 | 7.6 GB | [train.tar.gz](...) |
| sAlex | val | 553,218 | 408 MB | [val.tar.gz](...) |

### Data Access Example
```python
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
```

## 4) OMC25 (Open Molecular Crystals 2025)

The Open Molecular Crystals 2025 (OMC25) dataset was announced along with UMA, and comprises ~25 million calculations of organic molecular crystals from random packing of OE62 structures into various 3D unit cells. 

### Key Features:
- Calculated at the PBE+D3 level of theory via VASP
- Focuses on organic molecular crystals
- Generated from random packing configurations
- More details and download information coming soon
```
