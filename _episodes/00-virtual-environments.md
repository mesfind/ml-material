---
title: Virtual Environments
teaching: 1
exercises: 1
questions:
- "What is a virtual environment?"
- "How do virtual environments help manage dependencies?"
- "How do you create and activate a virtual environment?"
objectives:
- "Learn about virtual environments and their purpose in Python projects."
- "Understand how to create and manage virtual environments."
- "Get hands-on experience with setting up and using a virtual environment."
keypoints:
- "Virtual environments isolate dependencies."
- "Using Conda or venv for creating environments."
- "Activating and deactivating virtual environments."
---

# Python Environment Management

In ML for materials science, dependency isolation prevents conflicts between different projects.
**Conda** and Python’s built-in **venv** are both used to create environments with their own Python version and libraries. Conda is preferred here due to its ability to handle scientific and compiled packages.

---

## Conda Environments

A Conda environment is a self-contained directory with a specific Python version and packages.
Do not install packages into the base Conda environment — create one per project.

### Creating a Conda Environment for Materials Science ML (Linux/MacOS)

```bash
conda create --name MatML python=3.12
conda activate MatML
```

### Installing Core ML Libraries for Materials Science

```bash
conda install numpy pandas scikit-learn matplotlib seaborn jupyterlab
pip3 install torch torchvision pymatgen ase matminer  fairchem-core fairchem-data-oc fairchem-applications-cattsunami x3dase  m3gnet matgl chgnet mp-api 
```

### Exporting and Sharing Environments

```bash
conda env export --no-builds --file MatML.yaml
```

Recreate the environment elsewhere:

```bash
conda env create --file MatML.yaml
```

### Deactivating and Removing Environments

```bash
conda deactivate
conda env remove --name MatML
```

---

## Virtual Environments with `venv`

`venv` is built into Python and is lighter weight but requires manual installation of all dependencies with `pip`.

**Linux/MacOS**

```bash
python3 -m venv MatML
source MatGNN/bin/activate
pip3 install torch torchvision pymatgen ase matminer  fairchem-core fairchem-data-oc fairchem-applications-cattsunami x3dase  m3gnet matgl chgnet mp-api describe numpy pandas scikit-learn matplotlib seaborn jupyterlab ipython
```

**Windows PowerShell**

```powershell
python -m venv MatML
MatGNN\Scripts\activate
pip3 install torch torchvision pymatgen ase matminer  fairchem-core fairchem-data-oc fairchem-applications-cattsunami x3dase  m3gnet matgl chgnet mp-api describe numpy pandas scikit-learn matplotlib seaborn jupyterlab ipython
```

Deactivate with:

```bash
deactivate
```

---

## Exercise: Create a MatGNN Conda Environment with ML Libraries

1. **Create and activate environment**

```powershell
conda create --name MatML python=3.12
conda activate MatML
```

2. **Install ML libraries for materials science**

```powershell
conda install numpy pandas scikit-learn matplotlib seaborn jupyterlab
pip3 install torch torchvision pymatgen ase matminer  fairchem-core fairchem-data-oc fairchem-applications-cattsunami x3dase  m3gnet matgl chgnet mp-api describe
```

3. **Verify installation**

~~~
import pymatgen
import matminer
import fairchem.core
import m3gnet
import chgnet
import mp_api
import describe
import ase
print("Libraries loaded successfully")
~~~
{: .python}

4. **Export to YAML**

```powershell
conda env export --no-builds --file MatML.yaml
```

5. **Remove and recreate environment**

```powershell
conda deactivate
conda env remove --name MatML
conda env create --file MatML.yaml
```

---

## Ready-to-use `MatML.yaml`

```yaml
name: MatML
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyterlab
  - pip
  - pip3:
      - torch==2.3.1
      - torchvision
      - pymatgen
      - ase
      - matminer
      - fairchem-core
      - fairchem-data-oc
      - fairchem-applications-cattsunami
      - x3dase
      - dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html
      - m3gnet
      - matgl
      - chgnet
      - mp-api
      - describe
```

pip3 install torch pymatgen matminer fairchem-core  m3gnet matglchgnet mp-api describe numpy pandas scikit-learn matplotlib seaborn jupyterlab ipython

You can create this environment directly with:

```bash
conda env create --file MatML.yaml
conda activate MatML
```


