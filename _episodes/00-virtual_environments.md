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
conda create --name MatGNN python=3.10
conda activate MatGNN
```

### Installing Core ML Libraries for Materials Science

```bash
conda install numpy pandas scikit-learn matplotlib seaborn jupyterlab
pip install matminer fairchem megnet m3gnet chgnet mp-api
```

### Exporting and Sharing Environments

```bash
conda env export --no-builds --file MatGNN.yaml
```

Recreate the environment elsewhere:

```bash
conda env create --file MatGNN.yaml
```

### Deactivating and Removing Environments

```bash
conda deactivate
conda env remove --name MatGNN
```

---

## Virtual Environments with `venv`

`venv` is built into Python and is lighter weight but requires manual installation of all dependencies with `pip`.

**Linux/MacOS**

```bash
python3 -m venv MatGNN
source MatGNN/bin/activate
pip install matminer fairchem megnet m3gnet chgnet mp-api numpy pandas scikit-learn matplotlib seaborn jupyterlab
```

**Windows PowerShell**

```powershell
python -m venv MatGNN
MatGNN\Scripts\activate
pip install matminer fairchem megnet m3gnet chgnet mp-api numpy pandas scikit-learn matplotlib seaborn jupyterlab
```

Deactivate with:

```bash
deactivate
```

---

## Exercise: Create a MatGNN Conda Environment with ML Libraries

1. **Create and activate environment**

```powershell
conda create --name MatGNN python=3.10
conda activate MatGNN
```

2. **Install ML libraries for materials science**

```powershell
conda install numpy pandas scikit-learn matplotlib seaborn jupyterlab
pip install matminer fairchem megnet m3gnet chgnet mp-api
```

3. **Verify installation**

```python
import matminer
import fairchem
import megnet
import m3gnet
import chgnet
import mp_api
print("Libraries loaded successfully")
```

4. **Export to YAML**

```powershell
conda env export --no-builds --file MatGNN.yaml
```

5. **Remove and recreate environment**

```powershell
conda deactivate
conda env remove --name MatGNN
conda env create --file MatGNN.yaml
```

---

## Ready-to-use `MatGNN.yaml`

```yaml
name: MatGNN
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyterlab
  - pip
  - pip:
      - matminer
      - fairchem
      - megnet
      - m3gnet
      - chgnet
      - mp-api
```

Participants can create this environment directly with:

```bash
conda env create --file MatGNN.yaml
conda activate MatGNN
```


