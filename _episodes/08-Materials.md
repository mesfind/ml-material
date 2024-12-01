---
title: Machine Learning For Material Science
teaching: 1
exercises: 0
questions:
- "Can we have a comprehensive dataset consisting of large structures from various compounds spanning the whole periodic table"
- "Do we have an machine learning libary that able to study electron interactions and charge distribution in atomistic modeling with near DFT accuracy"
objectives:
- "Describe approaches for materials representation including chemical composition and crystal structure."
- "Discover structure and property information from public databases using Python."
- "Predict materials property using transfer learning"
keypoints:
- "Artificial intelligence and machine learning are being increasingly used in scientific domains such as computational science"
- "RDKit can be used to calculate various molecular properties for machine learning in materials science"
- "PyMatGen help to query structures from MP repository"
- "Descriptor-based ML approaches focus on using numerical features (descriptors) that summarize key aspects of the material"
- "Graph-based ML techniques are particularly suitable when dealing with materials that can be represented as networks or graphs"
- "In practice, hybrid approaches that combine aspects of both graph-based and descriptor-based methods are also gaining traction, leveraging the strengths of each depending on the task at hand."
- "Crystal Hamiltonian Graph neural Network is pretrained on the GGA/GGA+U static and relaxation trajectories from Materials Project"
- "Charge-informed molecular dynamics can be simulated with pretrained CHGNet through ASE python interface"
- "CHGNet can perform fast structure optimization and provide site-wise magnetic moments. This makes it ideal for pre-relaxation and MAGMOM initialization in spin-polarized DFT"
---

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>




This is a  lesson that cover the basics of using machine learning for material science. I will walk you through a few Python scripts that will enable you to classify materials and predict their properties.

The contents are targeted to beginers and scientists who have some experience with crystal structure (just the basics, like lattice constants and atomic positions). I also expect you to have basic experience with programming (in any language; Python, C, C++, C#, Java, JavaScript, etc).



# What is machine learning

So what to we use machine learning for in material science? The main purpose of machine learning here is the prediction of properties of a given material. This property can either be a class (the classification problem) or a quantity (the regression problem).



## The machine learning workflow

So how is this all done? Generally, we can think of machine learning as a 3-step process:
- Step A: First, find numerical/categoricall **descriptors** that can describe your material. That is: every material in your dataset should be uniquely represented by an array of numbers/categories.
- Step B: Then, apply your procedure to your entire dataset of structures to form a sheet of material descriptors vs. **target properties**.
- Step C: Use machine learning to predict the target properties based on the descriptors.



Before we go deeper into the details, we will need to learn a few things today:
- The programming language (python)
- The database of materials, from which we will get our data and create the dataset

Let's start!

## Google Colab

If you wish to learn to program in python and you don't have python installed on your computer, or you don't wish to struggle with the headache of setting it up, then you can use the Google Colab. This is a website that was developed by Google and is publicly available for anyone to use. It is pretty much an online python compiler, where you can write python code and run it, all on your web browser.

This tutorial, as you can see, is already running on Google Colab (let's just call it Colab from now on). On Colab, you can create python **notebooks**, which are known as Jupyter notebooks. Jupyter is a programming environment for python that allows the programmer to write documents, just like this one, where there is both text and code. To demonstrate the placement of code here, check out the below section. This is some python code that prints out the text `"Hello Computational Data Science!"`. Press the play button, and it will run that code, and actually print out `"Hello Computational Data Science!"`.


~~~
print("Hello Computational Data Science! Happy you're in HU!")
~~~
{: .python}

~~~
Hello Computational Data Science! Happy you're in HU!
~~~
{: .output}


## Quick Overview on Python 

### Basic Python: Variables and Operations

Python is an excellent and accessible language for computational tasks in drug discovery. Its simplicity and flexibility make it an ideal tool for researchers. Python code can be executed on various platforms, including laptops, desktops, and even mobile devices, thanks to numerous online Python servers that facilitate code execution without the need for local installations.

### The Print Statement

Let’s begin with a fundamental operation: using Python to print text. Consider the scenario where we want the program to display `"Initiating compound screening protocol"` on the screen. This can be achieved using the `print` statement, which instructs the computer to display a specific text, known as a *string*, on the screen.

Here's an example of how this works:

~~~
print("Initiating compound screening protocol")
~~~
{: .python}

~~~
Initiating compound screening protocol
~~~
{: .output}

Notice the simplicity of the above line of code. It’s an executable line, meaning that when executed, the program will run and display the specified message. This can be done seamlessly on platforms like Google Colab, which allows Python code to be executed directly in the browser.

What exactly happens here? We utilized the `print` function, which takes the string `"Initiating compound screening protocol"` and displays it. This function is essential for providing outputs during various stages of drug discovery computations.

Let’s try printing two messages sequentially:

~~~
print("Screening initiated")
print("Analyzing compound interaction with target proteins")
~~~
{: .python}

~~~
Screening initiated
Analyzing compound interaction with target proteins
~~~
{: .output}

### Python Data Types: Strings, Numbers, and Boolean Values

Python supports various data types that are crucial for different operations in drug discovery. Here, we will explore three fundamental data types: strings, numbers, and boolean values.

Numbers are straightforward in Python. If you input a number, say `5`, into the interpreter, it will return `5`.

### Variables

In Python, variables are used to store values in memory, facilitating complex computations and data handling. For instance, we can store a string that represents a step in our drug discovery protocol:

~~~
protocol_step = "Identifying potential drug candidates"
print(protocol_step)
~~~
{: .python}

~~~
Identifying potential drug candidates
~~~
{: .output}

This simple variable assignment enables us to store and retrieve data efficiently, a fundamental requirement in managing extensive drug discovery workflows.




### Arithmetic Operations: `+`, `-`, `*`, `/`, `%`, `//`

In material science computations, arithmetic operations are frequently used for various calculations. Python supports standard arithmetic operators such as `+` (addition), `-` (subtraction), `*` (multiplication), `/` (division), and `%` (modulus). The `//` operator, known as the floor division operator, provides the quotient of a division, discarding any fractional part. For instance, `9 // 4` results in `2`.

The `%` operator, or modulus, returns the remainder of a division. In the example `9 / 4 = 2 + 1/4`, the quotient is `2` and the remainder is `1`. Thus, in Python, `9 % 4` equals `1`.

### Comparison Operations: `==`, `!=`, `<`, `>`, `<=`, `>=`

In material science, comparing values is essential for data analysis and simulations. Python provides comparison operators that yield boolean values (True or False) based on the relationship between variables:

- `a == b` checks if `a` is exactly equal to `b`.
- `a != b` checks if `a` is not equal to `b`.
- `a < b` checks if `a` is less than `b`.
- `a > b` checks if `a` is greater than `b`.
- `a <= b` checks if `a` is less than or equal to `b`.
- `a >= b` checks if `a` is greater than or equal to `b`.

Here’s an example demonstrating these operations:

~~~
# This code demonstrates comparison operations 
a = 7.5  # represents a property value of a material
b = 9.8  # represents another property value of a material

c = a == b
print("Is a equal to b?", c)

d = a > b
print("Is a greater than b?", d)

e = a <= b
print("Is a less than or equal to b?", e)
~~~
{: .python}

~~~
Is a equal to b? False
Is a greater than b? False
Is a less than or equal to b? True
~~~
{: .output}

In material science, these operations could be used to compare properties such as tensile strength, thermal conductivity, or electrical resistance between different materials or samples. This enables researchers to make informed decisions based on quantitative data analysis.


### Lists

A `list` in python is exactly what its name suggests, a list of things. Like a list of numbers, names, or even a mix of both. To create a list, we have to follow a simple syntax rule: enclose the things in the list between two *square brackets*, like those `[` and `]`, and separate between the list elements using commas, `,`. So for example, here is a list of numbers: `a = [4,6,7,1,0]`, a list of strings: `a = ["a","?","neptune is a planet"]`, a list of both: `a = [3,0,"Where is my car?"]`.

Well, you can also create a list of lists in python! And you can *nest* as many lists as you want. Here is an example: `a = [[1,2],[3,4],[5,6]]`. This is a list of three elements, each element being itself a list of two elements.

### Accessing and Modifying Elements in Arrays

In the context of material science data analysis, accessing and modifying elements within an array is a fundamental task. Here is how you can achieve this:

#### Accessing Array Elements

To access an element in an array, determine its *index* and use square brackets for retrieval. The index is an integer, starting from `0` for the first element. Consider the following example:
~~~
data = ["Sample_A", "Fe", "Cu", "Analysis", "Complete", 4.5, "Data", "Confirmed"]
print(data[0])
~~~
{: .python}

~~~
Sample_A
~~~
{: .output}

Here, the string `'Sample_A'` is the first element of the array `data`, hence its index is `0`, accessible via `data[0]`.

#### Modifying Array Elements

To modify an element in an array, simply assign a new value to the specific index.

~~~
data = ["Sample_A", "Fe", "Cu", "Analysis", "Complete", 4.5, "Data", "Confirmed"]
data[5] = "Verified"
print(data)
~~~
{: .python}

~~~
['Sample_A', 'Fe', 'Cu', 'Analysis', 'Complete', 'Verified', 'Data', 'Confirmed']
~~~
{: .output}

### Tuples

A tuple is similar to an array but with a critical distinction: tuples are immutable. Once created, the elements within a tuple cannot be changed. Tuples are defined using parentheses `(` and `)`.

For example, let's create a tuple and try to modify one of its elements:

```python
experiment_results = (7.1, 8.3, 5.4)
#experiment_results[0] = 6.5
#The above line will result in an error as tuples do not support item assignment!
```
{: .python}

Attempting to execute the commented line will raise an error since tuples are immutable, making them ideal for storing constant data that should not change throughout the analysis process.


### Dictionaries

We learned in lists and tuples that the elements are indexed. The index is an integer that starts from `0`. A dictionary extends the indexing concept: a dictionary is a collection of indexed objects, where the indices themselves can be anything *immutable*: numbers, float, strings and tuples (and frozensets, but we won't discuss that one today).

The syntax for creating a dictionary is as follows: `{key:value}`, where `key` is the index, `value` is any data type. For example,


~~~
a = {'C': 6, 'O': 8, 'Fe':26}
print(a['C'])
b = {'a':"lists",'b':"tuples",'c':"sets",'d':"dictionaries"}
print(b['c'])
~~~
{: .python}


~~~
6
sets

~~~
{: .output}


### Python libraries


One of the most powerful features of python is its libraries. A library is a python script that was written by someone, and that can perform a set of tasks. You can make use of a python library by just using the `import` command. For example, when you want to calculate the logarithm, the `log()` function you would look for exists in the `numpy` library.

~~~
import numpy as np
print(np.log(11))
~~~
{: .python}

~~~
2.3978952727983707
~~~
{: .output}


### Installing Python Libraries for Material Science

Using Python for material science is made easier with various libraries tailored for analysis and simulations. Here's how to install them:

- **pip**:
  - Pip is the default package manager for Python, used to install packages from the Python Package Index (PyPI).
  - It works well with Python’s virtual environments, allowing you to create isolated project environments.
  - To install a package with pip, use the command `pip install package-name`.
  - To update pip itself, use the command !pip install pip -U --root-user-action=ignore

- **Conda**:
  - Conda can install packages for multiple programming languages, including Python, R, Ruby, and JavaScript.
  - It provides pre-compiled packages, which makes installations faster.
  - Conda has its own version of pip for installing Python packages within Conda environments.
  - To install a package with Conda, use the command `conda install package-name`.


  ~~~
  !pip install pip -U --root-user-action=ignore # update the pip on notebook
  ~~~
  {: .python}



A DataFrame organizes data into a 2-dimensional table of rows and columns, much like a spreadsheet. They are useful tools to store, access, and modify large sets of data.

### Example: Creating and Modifying a DataFrame

First, we'll import the `pandas` library and create a DataFrame with some elemental data:

~~~
import pandas as pd  # Data manipulation using DataFrames

data = {
    "Element": ['C', 'O', 'Fe', 'Mg', 'Xe'],
    "Atomic_Number": [6, 8, 26, 12, 54],
    "Atomic_Mass": [12, 16, 56, 24, 131]
}

# Loading data into DataFrame
df = pd.DataFrame(data)

# Setting the 'Element' column as the index
df = df.set_index("Element")
df.head()
~~~
{: .python}

The output is:

~~~
         Atomic_Number  Atomic_Mass
Element                            
C                    6           12
O                    8           16
Fe                  26           56
Mg                  12           24
Xe                  54          131
~~~
{: .output}

### Adding a New Column

Adding a new column to a DataFrame is a common operation to enrich your dataset with additional information.

~~~
df["Energy(eV)"] = [5.47, 5.14, 0.12, 4.34, 7.01]

print(df["Energy(eV)"])
~~~
{: .python}

The output is:

~~~
Element
C     5.47
O     5.14
Fe    0.12
Mg    4.34
Xe    7.01
Name: Energy(eV), dtype: float64
~~~
{: .output}

## Writing a Function

The Arrhenius equation is a fundamental formula in physical chemistry that describes how the rate of a chemical reaction depends on temperature and activation energy. The equation is given by:

\[ k = D_0 \cdot e^{-\frac{E_a}{k_B T}} \]

where:

- \( k \) is the rate constant.
- \( D_0 \) is the pre-exponential factor (default is set to 1).
- \( E_a \) is the activation energy (in eV).
- \( T \) is the temperature (in Kelvin).
- \( k_B \) is the Boltzmann constant.

Let's write the Arrhenius equation as a Python function using NumPy and SciPy:

~~~
import numpy as np
from scipy.constants import physical_constants

# Define constants
k_B = physical_constants['Boltzmann constant in eV/K'][0]

# Arrhenius function
def arrhenius(activation_energy, temperature, D0=1):
    """
    Calculates the rate using the Arrhenius equation.
    
    Parameters:
    activation_energy (float): the activation energy in eV.
    temperature (float): the temperature in K (must be > 0).
    D0 (float): the pre-exponential factor (default is 1).
    
    Returns:
    float: the rate of the reaction.
    """
    if np.any(temperature <= 0):
        raise ValueError("Temperature must be greater than 0 K.")
    return D0 * np.exp(-activation_energy / (k_B * temperature))
~~~
{: .python}

This function calculates the reaction rate using the Arrhenius equation, ensuring the temperature is valid.

### Applying the Function

Now, let's calculate the reaction rates for each element at a specific temperature (e.g., 300 K):

~~~
# Define a temperature for the calculation
temperature = 300  # Example: 300 K

# Calculate the reaction rates for each material
df["Reaction_Rate"] = df["Energy(eV)"].apply(lambda x: arrhenius(x, temperature))

# Print the updated DataFrame
print(df)
~~~
{: .python}

The output is:

~~~
         Atomic_Number  Atomic_Mass  Energy(eV)  Reaction_Rate
Element                                                       
C                    6           12        5.47   1.282462e-92
O                    8           16        5.14   4.485392e-87
Fe                  26           56        0.12   9.640260e-03
Mg                  12           24        4.34   1.233698e-73
Xe                  54          131        7.01  1.726566e-118
~~~
{: .output}

### Plotting the Results

Finally, let's plot the reaction rates over a range of temperatures using Matplotlib:

~~~
import matplotlib.pyplot as plt

# Pre-exponential term in cm^2/s
D0 = 0.5

# Temperature range in K
T = np.linspace(1, 1000, 100)

# Example activation energy in eV
activation_energy = 0.83

# Calculate rates
rates = arrhenius(activation_energy, T, D0)

# Plotting
plt.figure(figsize=(5, 3))
plt.plot(T, rates, label=f'Activation Energy = {activation_energy} eV')
plt.xlabel('Temperature (K)')
plt.ylabel('$D_{ion}$ (cm$^2$/s)')  # Adding units to y-axis
plt.title('Thermally Activated Transport')
plt.legend()
plt.grid(True)
plt.show()
~~~
{: .python}

This plot illustrates how the reaction rate varies with temperature, highlighting the thermally activated nature of the process.



In computational chemistry and cheminformatics, chemical compounds can be represented using different notation systems. Two common notation systems are SMILES (Simplified Molecular Input Line Entry System) and IUPAC (International Union of Pure and Applied Chemistry) names. SMILES notation is a way to describe a chemical structure using a line of text, which is compact and easy for computers to process. On the other hand, IUPAC names are standardized chemical names that are more descriptive and human-readable but can be quite complex.

To facilitate the conversion between these two notation systems, we can use a specialized library called rdkit. The rdkit library provides functions to translate chemical compounds from SMILES notation to IUPAC names and vice versa. Below is an example of how to use RDKit to perform these translations:

1. __SMILES to IUPAC Name Translation__: Using RDKit, we can convert a SMILES string to a molecule object and then use an external library (like PubChemPy) to fetch the IUPAC name.
2. __IUPAC Name to SMILES Translation__: Similarly, we can use RDKit to convert an IUPAC name to a SMILES string by searching a chemical database.


~~~
!pip install rdkit
!pip install pubchempy
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp

# SMILES to IUPAC name translation using RDKit and PubChemPy
SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
molecule = Chem.MolFromSmiles(SMILES)
iupac_name = pcp.get_compounds(SMILES, 'smiles')[0].iupac_name
print("IUPAC name of " + SMILES + " is: " + iupac_name)

# IUPAC name to SMILES translation using RDKit and PubChemPy
IUPAC_name = "1,3,7-trimethylpurine-2,6-dione"
compound = pcp.get_compounds(IUPAC_name, 'name')[0]
smiles = compound.canonical_smiles
print("SMILES of " + IUPAC_name + " is: " + smiles)

~~~
{: .python}




### Molecular Property Calculations 

RDKit can be used to calculate various molecular properties such as molecular weight, logP (octanol-water partition coefficient), and the number of hydrogen bond donors and acceptors.


~~~
from rdkit import Chem
from rdkit.Chem import Descriptors

# SMILES string of a molecule
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

# Convert SMILES to a molecule object
molecule = Chem.MolFromSmiles(smiles)

# Calculate molecular properties
molecular_weight = Descriptors.MolWt(molecule)
logp = Descriptors.MolLogP(molecule)
num_h_donors = Descriptors.NumHDonors(molecule)
num_h_acceptors = Descriptors.NumHAcceptors(molecule)

print(f"Molecular weight of {smiles}: {molecular_weight} g/mol")
print(f"LogP of {smiles}: {logp}")
print(f"Number of hydrogen bond donors in {smiles}: {num_h_donors}")
print(f"Number of hydrogen bond acceptors in {smiles}: {num_h_acceptors}")
~~~
{: .python}


~~~
Molecular weight of CN1C=NC2=C1C(=O)N(C(=O)N2C)C: 194.194 g/mol
LogP of CN1C=NC2=C1C(=O)N(C(=O)N2C)C: -1.0293
Number of hydrogen bond donors in CN1C=NC2=C1C(=O)N(C(=O)N2C)C: 0
Number of hydrogen bond acceptors in CN1C=NC2=C1C(=O)N(C(=O)N2C)C: 6
~~~
{: .output}


## Substructure Searching
RDKit can be used to search for specific substructures within a larger molecule.



~~~
from rdkit import Chem

# SMILES strings of molecules
smiles_list = ["CCO", "CCN", "CC(C)O", "CC(C)C"]

# Convert SMILES to molecule objects
molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

# Define the substructure to search for (ethanol)
substructure = Chem.MolFromSmiles("CCO")

# Perform substructure search
matches = [mol.HasSubstructMatch(substructure) for mol in molecules]

for smiles, match in zip(smiles_list, matches):
    print(f"Does {smiles} contain the substructure CCO? {'Yes' if match else 'No'}")
~~~
{: .python}


~~~
Does CCO contain the substructure CCO? Yes
Does CCN contain the substructure CCO? No
Does CC(C)O contain the substructure CCO? Yes
Does CC(C)C contain the substructure CCO? No
~~~
{: .output}


### Molecular Visualization
RDKit can be used to visualize molecules. For this example, we'll use RDKit's built-in drawing capabilities to display a molecule.

~~~
from rdkit import Chem
from rdkit.Chem import Draw

# SMILES string of a molecule
smiles = "CCO"  # Ethanol

# Convert SMILES to a molecule object
molecule = Chem.MolFromSmiles(smiles)

# Draw the molecule
img = Draw.MolToImage(molecule)
img.show()

~~~
{: .python}



### Converting Between Different Formats
RDKit can convert molecules between various file formats, such as SMILES, InChI, and Molfile.

~~~
from rdkit import Chem

# SMILES string of a molecule
smiles = "CCO"  # Ethanol

# Convert SMILES to a molecule object
molecule = Chem.MolFromSmiles(smiles)

# Convert molecule to InChI
inchi = Chem.MolToInchi(molecule)
print(f"InChI of {smiles}: {inchi}")

# Convert molecule to Molfile
molfile = Chem.MolToMolBlock(molecule)
print(f"Molfile of {smiles}:\n{molfile}")
~~~
{: .python}

~~~
InChI of CN1C=NC2=C1C(=O)N(C(=O)N2C)C: InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3
Molfile of CN1C=NC2=C1C(=O)N(C(=O)N2C)C:

     RDKit          2D

 14 15  0  0  0  0  0  0  0  0999 V2000
    2.7760    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2760    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.3943    1.2135    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0323    0.7500    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0323   -0.7500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.3943   -1.2135    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7062   -2.6807    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1328   -3.1443    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4086   -3.6844    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8351   -3.2209    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.9499   -4.2246    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1470   -1.7537    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -3.5736   -1.2902    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0967   -5.1517    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  7  1  0
  7  8  2  0
  7  9  1  0
  9 10  1  0
 10 11  2  0
 10 12  1  0
 12 13  1  0
  9 14  1  0
  6  2  1  0
 12  5  1  0
M  END
~~~
{: .output}


### Converting a SMILES string into a 3D structure

 Converting a SMILES string into a 3D structure and visualizing it involves generating 3D coordinates for the molecule and then using a visualization library to display it. We'll use RDKit to generate the 3D coordinates and py3Dmol for visualization

~~~
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

# SMILES string of a molecule
smiles = "CCO"  # Ethanol

# Convert SMILES to a molecule object
molecule = Chem.MolFromSmiles(smiles)

# Add hydrogen atoms
molecule = Chem.AddHs(molecule)

# Generate 3D coordinates
AllChem.EmbedMolecule(molecule)
AllChem.UFFOptimizeMolecule(molecule)

# Convert the molecule to a 3D format compatible with py3Dmol
block = Chem.MolToMolBlock(molecule)

# Visualize the molecule using py3Dmol
view = py3Dmol.view(width=400, height=300)
view.addModel(block, 'mol')
view.setStyle({'stick': {}})
view.setBackgroundColor('white')
view.zoomTo()
view.show()

~~~
{: .python}


### Generating Morgan Fingerprints (Circular Fingerprints)

Morgan fingerprints are widely used in cheminformatics and are also known as Extended-Connectivity Fingerprints (ECFP).

~~~
## Feature vectors for machine learning
from rdkit import Chem
from rdkit.Chem import AllChem

# SMILES string of a molecule
smiles = "CCO"  # Ethanol
# Convert SMILES to a molecule object
molecule = Chem.MolFromSmiles(smiles)
# Generate Morgan fingerprint
radius = 2
n_bits = 1024
morgan_fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits)
# Convert to a list of integers (feature vector)
feature_vector = list(morgan_fp)
print("Morgan Fingerprint Feature Vector:", feature_vector)
~~~
{: .python}

~~~
Morgan Fingerprint Feature Vector: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
~~~
{: .output}


### Generating Feature Vectors for a Dataset
Here is an example of generating feature vectors for a dataset of molecules:


~~~
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# List of SMILES strings
smiles_list = ["CCO", "CCN", "CC(C)O", "CC(C)C"]

# Function to generate Morgan fingerprint feature vector
def generate_morgan_fp(smiles, radius=2, n_bits=1024):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    # Create a Morgan generator
    morgan_generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    morgan_fp = morgan_generator.GetFingerprint(molecule)
    
    # Create a fixed-length feature vector
    feature_vector = [0] * n_bits
    for bit in morgan_fp.GetOnBits():
        feature_vector[bit] = 1
    
    return feature_vector

# Generate feature vectors for the dataset
feature_vectors = [generate_morgan_fp(smiles) for smiles in smiles_list]

# Create a DataFrame
df = pd.DataFrame(feature_vectors, columns=[f'Bit_{i}' for i in range(len(feature_vectors[0]))])
df['SMILES'] = smiles_list

print(df)
~~~
{: .python}

~~~
 Bit_0  Bit_1  Bit_2  Bit_3  Bit_4  Bit_5  Bit_6  Bit_7  Bit_8  Bit_9  ...  \
0      0      0      0      0      0      0      0      0      0      0  ...   
1      0      0      0      0      0      0      0      0      0      0  ...   
2      0      1      0      0      0      0      0      0      0      0  ...   
3      0      1      0      0      0      0      0      0      0      0  ...   

   Bit_1015  Bit_1016  Bit_1017  Bit_1018  Bit_1019  Bit_1020  Bit_1021  \
0         0         0         0         0         0         0         0   
1         0         0         0         0         0         0         0   
2         0         0         0         0         0         0         0   
3         0         0         0         0         0         0         0   

   Bit_1022  Bit_1023  SMILES  
0         0         0     CCO  
1         0         0     CCN  
2         0         0  CC(C)O  
3         0         0  CC(C)C  

[4 rows x 1025 columns]
~~~
{: .output}


## The MaterialsProject(MP) database

The MaterialsProject (MP) database is a massive amount of material science data that was generated using density functional theory (DFT). Have a look at the database here: https://materialsproject.org. Check the statistics at the bottom of the page. There are 124,000 inorganic crystals in the database, along with the DFT-calculated properties of these materials. There are also 530,000 nanoporous materials, as well as other stuff. It's a huge amount of material data.

### Signing up and the API key

You will need to **sign up** in materialsproject.org to be able to access the database. Signing up there is free.

Once you sign up, you can obtain an **API key** that will enable you to access the database using python. Will discuss this further shortly.


### A look at MaterialsProject

Let's have a look at the 124,000 inorganic crystals data. Each one of these crystals is a 3D crystal that includes: a number of elements, arranged in a lattice structure. Check, for example, the MP page for diamond: https://materialsproject.org/materials/mp-18767/.



Note that each material on MP is identified by an ID that goes like `mp-X` where `X` is a number. The ID of diamond is `mp-18767`. People use these identifiers when referring to MP materials in papers, and we will use them soon when we start querying materials from MP using python.

There you will find the crystal structure, the lattice parameters, the basic properties (in a column to the right of the figure that displays the crystal), and then a range of DFT-calculated properties.

### The DFT properties

These are quantities that are calculated for each crystal in MP. In fact, every thing you see on the MP page for diamond was calculated using DFT. 

- For a given elemental composition, the lattice parameters and the positions of the atoms within the lattice are all obtained using DFT.
- For the obtained crystal structure, the `Final Magnetic Moment`, `Formation Energy / Atom`, `Energy Above Hull / Atom`, `Band Gap` are calculated. The `Density` is derived from the obtained crystal structure.
- Further DFT calculations are performed to obtain the band structure as well as other properties that you can find as you scroll down the structure page on MP.

Some of the crystals on MP correspond to crystals that exist in nature, and some are purely hypothetical. The hypothetical crystals have been generated by some algorithm that uses artificial intelligence, or probably by simple elemental substitution.

### Why is MP great for machine learning?

Because of the huge amount of materials (dataset) and DFT-calculated properties (target properties). That much data can be utilised using a range of machine learning methods to make predictions with various levels of accuracy.

## The PyMatGen python library

To be able to query the MP database, the MP team provided the community with a python library that you can install on your computer (or in Colab as I will show you).

Remember: to be able to run the codes in this section, you must obtain an API key from the MP website: https://materialsproject.org/docs/api#API_keys.

The first thing we do here is to install PyMatGen in Colab.

~~~
!pip install --upgrade -q pip 
!pip install --upgrade -q pymatgen scipy 
~~~
{: .python}


## Structure file formats

One of the most common file formats that describe crystal structure is the CIF format (Crystallographic Information File). The official definition of this formnat is here: https://www.iucr.org/resources/cif.


But we are not going to learn the details of the format. We will just learn how to open a CIF with pythton. Here is how we can do this.


~~~
from pymatgen.io.cif import CifParser
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

with MPRester(api_key="Your_API_Key") as mpr:
    # Retrieve the crystal structures for a list of materials
    docs = mpr.materials.summary.search(material_ids=["mp-18767"], fields=["material_id", "structure"])

    # Save the CIF files
    for doc in docs:
        material_id = doc.material_id
        structure = doc.structure
        filename = f"{material_id}.cif"
        
        # Convert the structure to CIF format and save
        cif = CifWriter(structure)
        cif.write_file(filename)
        
        print(f"Saved {filename}")
~~~
{: .python}

 computer, download the MP file mp-18767.cif and place it in the same director as your python console/script

 ~~~
 parser = CifParser('mp-18767.cif')
 ~~~
 {: .python}


In the above code, we imported a **class** from the PyMatGen library: the `CifParser` class. It allows us to create a new CIF file **object**. This object will then represent the CIF structure, and can be used to access its information.

Next, let's extract some information from the `CifParser` object.

~~~
structure = parser.get_structures()
# Returns a list of Structure objects
# #http://pymatgen.org/_modules/pymatgen/core/structure.html
# Let's print the first (and only) Structure object
print(structure[0])
~~~
{: .python}


~~~
Full Formula (Li2 Mn2 O4)
Reduced Formula: LiMnO2
abc   :   2.806775   4.603051   5.701790
angles:  90.000005  90.000003  90.000000
pbc   :       True       True       True
Sites (8)
  #  SP           a         b         c
---  ----  --------  --------  --------
  0  Li    0.749999  0.75      0.880615
  1  Li    0.249999  0.249999  0.119384
  2  Mn    0.749997  0.75      0.362786
  3  Mn    0.249996  0.249999  0.637214
  4  O     0.75      0.249999  0.862143
  5  O     0.25      0.75      0.599339
  6  O     0.749999  0.25      0.40066
  7  O     0.25      0.75      0.137858
~~~
{: .output}



~~~
LiMnO2 = structure[0]

print(LiMnO2.lattice)
print(LiMnO2.species)
print(LiMnO2.charge)
print(LiMnO2.cart_coords)
print(LiMnO2.atomic_numbers)
#print(LiMnO2.distance_matrix)
~~~
{: .python}

~~~
2.806775 0.000000 -0.000000
-0.000000 4.603051 -0.000000
0.000000 0.000000 5.701790
[Element Li, Element Li, Element Mn, Element Mn, Element O, Element O, Element O, Element O]
0.0
[[2.1051 3.4523 5.0211]
 [0.7017 1.1508 0.6807]
 [2.1051 3.4523 2.0685]
 [0.7017 1.1508 3.6333]
 [2.1051 1.1508 4.9158]
 [0.7017 3.4523 3.4173]
 [2.1051 1.1508 2.2845]
 [0.7017 3.4523 0.786 ]]
(3, 3, 25, 25, 8, 8, 8, 8)
~~~
{: .output}


### Querying structures using PyMatGen
Now let's use PyMatGen to query structures from MP. To be able to do that, we need first to create a MPRester with the API key that we receive from MP.

~~~
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

with MPRester(api_key="Your_API_Key") as mpr:
    # Retrieve the crystal structures for a list of materials
    docs = mpr.materials.summary.search(material_ids=["mp-18767"], fields=["material_id", "structure"])

    # Save the CIF files
    for doc in docs:
        material_id = doc.material_id
        structure = doc.structure
        filename = f"{material_id}.cif"
        
        # Convert the structure to CIF format and save
        cif = CifWriter(structure)
        cif.write_file(filename)
        
        print(f"Saved {filename}")
~~~
{: .python}

~~~
Saved mp-18767.cif
~~~
{: .output}


Then we can use the object variable, to access MP. For today, I willl just show you a simple query

~~~
parser = CifParser('mp-18767.cif')
structure = parser.parse_structures(primitive=True)
LiMnO2 = structure[0]
LiMnO2
~~~
{: .python}

~~~
Structure Summary
Lattice
    abc : 2.80677545 4.603050630000001 5.70178977
 angles : 90.00000455 90.00000253 90.0
 volume : 73.6655815136019
      A : 2.8067754499999973 0.0 -1.2393830676176043e-07
      B : -1.663517649842935e-14 4.603050629999986 -3.6553967057842006e-07
      C : 0.0 0.0 5.70178977
    pbc : True True True
PeriodicSite: Li0 (Li) (2.105, 3.452, 5.021) [0.75, 0.75, 0.8806]
PeriodicSite: Li1 (Li) (0.7017, 1.151, 0.6807) [0.25, 0.25, 0.1194]
PeriodicSite: Mn2 (Mn) (2.105, 3.452, 2.069) [0.75, 0.75, 0.3628]
PeriodicSite: Mn3 (Mn) (0.7017, 1.151, 3.633) [0.25, 0.25, 0.6372]
PeriodicSite: O4 (O) (2.105, 1.151, 4.916) [0.75, 0.25, 0.8621]
PeriodicSite: O5 (O) (0.7017, 3.452, 3.417) [0.25, 0.75, 0.5993]
PeriodicSite: O6 (O) (2.105, 1.151, 2.284) [0.75, 0.25, 0.4007]
PeriodicSite: O7 (O) (0.7017, 3.452, 0.786) [0.25, 0.75, 0.1379]
~~~
{: .output}



# Part 2 : Generating Descriptors for ML


## Descriptors

Before we start ML, let’s address a very important question that lies at the centre of the field of ML-driven material discovery: **how do we apply ML to predict crystal properties?**

## Building a simple descriptor vector for crystals

A possible solution to this problem is to use some statistics of atomic properties as the descriptor vector. For example:

- Average of the atomic numbers of all the elements in the crystal. For example, in silicon carbide, SiC, the average value would be the average of 14 (for Si) and 6 (for C), which is (14 + 6)/2 = 10. So that's now one number in the descriptor vector.
- The average of the ionization potential of the atoms
- The average of the electron affinity of the atoms
- And more averages

So we can keep adding averages of properties to this list, to expand the descriptor vector. This vector will not suffer from the lack of invariance issue pointed out above, because these are average values of quantities that do not depend on the geometry of the crystal.

Average is just one statistic. We can also add other statistics, such as the standard deviation and the variance. Adding those will triple the number elements in the descriptor vector above.


~~~
import numpy as np
from pymatgen.io.cif import CifParser
parser = CifParser('mp-18767.cif')
structure = parser.parse_structures(primitive=True)
LiMnO2 = structure[0]
mean_atomic_number=np.mean(LiMnO2.atomic_numbers)
max_atomic_number=np.max(LiMnO2.atomic_numbers)
min_atomic_number=np.min(LiMnO2.atomic_numbers)
std_atomic_number=np.std(LiMnO2.atomic_numbers)
print(mean_atomic_number,max_atomic_number,min_atomic_number,std_atomic_number)

~~~
{: .python}


~~~
11.0 25 3 8.336666000266533
~~~
{: .output}


However, there is a problem. A lot of materials exist in various phases. That is, for the same atomic composition, let's say SiC, there are several possible structures. Right now, there are 27 possible structures for SiC on MaterialsProject.org.
So, the above descriptors won't work. For example, for the case of SiC, all of the 27 SiC phases in MP will have the same values for the statistical values above.
To solve this problem, we have to add descriptors based on the geometrical arrangement of atoms. A simple such descriptor is to average the bond lengths (a bond is formed between two atoms).

~~~
mean_distance_matrix = np.mean(LiMnO2.distance_matrix)
max_distance_matrix = np.max(LiMnO2.distance_matrix)
min_distance_matrix = np.min(LiMnO2.distance_matrix)
std_distance_matrix = np.std(LiMnO2.distance_matrix)

print(mean_distance_matrix, max_distance_matrix,
      min_distance_matrix, std_distance_matrix)
~~~
{: .python}

~~~
2.3405195829366106 3.6611097462427473 0.0 1.0309531409177013
~~~
{: .output}


## Building a data set

Now it's time to build our dataset, before we can do machine learning on it. We will do this in 3 steps:

- Step 1: collecting the structures
- Step 2: pre-processing the data

### Step 1: Collecting the structures

We want to predict the bandgaps of structures, so we need to collect the structures (dataset) along with their corresponding bandgaps (target vector).

For this exercise, let's focus on stoichiometric perovskites: these are materials of the form ABC3. The followiing query will collect the CIFs and bandgaps for these materials from MP.


~~~
from mp_api.client import MPRester

# Replace 'YOUR_API_KEY' with your actual Materials Project API key
with MPRester("YOUR_API_KEY") as mpr:
    results = mpr.summary.search(formula="ABC3", fields=["structure", "band_gap"])

# Process the results
for doc in results:   
    # If you need the CIF, you can get it from the structure
    cif = doc.structure.to(fmt="cif")
~~~
{: .python}

### Step 2: Pre-processing the data

Here we will extract the data we need from the structures, put them in a pandas DataFrame and then apply normalization.

~~~

# We will calculate descriptor vector components based on the following prorperties:
# The atomic numbers of elemenst, the distances between the elements, and the 6 lattice parameters
# For the atomic numbers, we can just obtain their statistics. We will be gathering these statistics into python lists..
# Those lists should be initialized as empty list
mean_atomic_numbers = []
max_atomic_numbers = []
min_atomic_numbers = []
std_atomic_numbers = []
# Lattice parameters:
a_parameters = []
b_parameters = []
c_parameters = []
alpha_parameters = []
beta_parameters = []
gamma_parameters = []
# As for the interatomic distances: these are readily obatined via the attribute `distance_matrix`. Let's calcuale the statistics of this matrix too.
# Again, we can obtain statistics for the values in matrix
mean_distance_matrix = []
max_distance_matrix = []
min_distance_matrix = []
std_distance_matrix = []
# We also need to collect the target vector, or bandgaps

band_gaps = []

# The lattice decriptors can be obtained via the Lattice object in PyMatGen as follows:

print('a=',LiMnO2.lattice.abc[0],'b=',LiMnO2.lattice.abc[1],'c=',LiMnO2.lattice.abc[2])
print('alpha=',LiMnO2.lattice.angles[0],'beta=',LiMnO2.lattice.angles[1],'gamma=',LiMnO2.lattice.angles[2])

~~~
{: .python}

~~~
a= 2.80677545 b= 4.603050630000001 c= 5.70178977
alpha= 90.00000455 beta= 90.00000253 gamma= 90.0
~~~
{: .output}


Now let's iterate through the list of results and extract our descriptors into the above lists. This will take a few minutes.

~~~
import numpy as np
from pymatgen.io.cif import CifParser

# Initialize lists to store the data
mean_atomic_numbers, max_atomic_numbers, min_atomic_numbers, std_atomic_numbers = [], [], [], []
a_parameters, b_parameters, c_parameters = [], [], []
alpha_parameters, beta_parameters, gamma_parameters = [], [], []
mean_distance_matrix, max_distance_matrix, min_distance_matrix, std_distance_matrix = [], [], [], []
band_gaps = []

for doc in results:
    # Get the structure directly from the doc
    structure = doc.structure
    
    # Get the band gap
    bg = doc.band_gap
    
    # Calculate atomic number statistics
    atomic_numbers = structure.atomic_numbers
    mean_atomic_numbers.append(np.mean(atomic_numbers))
    max_atomic_numbers.append(np.max(atomic_numbers))
    min_atomic_numbers.append(np.min(atomic_numbers))
    std_atomic_numbers.append(np.std(atomic_numbers))
    
    # Lattice parameters
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    
    a_parameters.append(a)
    b_parameters.append(b)
    c_parameters.append(c)
    alpha_parameters.append(alpha)
    beta_parameters.append(beta)
    gamma_parameters.append(gamma)
    
    # Distance matrix statistics
    distance_matrix = structure.distance_matrix
    mean_distance_matrix.append(np.mean(distance_matrix))
    max_distance_matrix.append(np.max(distance_matrix))
    min_distance_matrix.append(np.min(distance_matrix))
    std_distance_matrix.append(np.std(distance_matrix))
    
    # Band gap
    band_gaps.append(bg)

~~~
{: .python}

Now you can use these lists for further processing or analysis. But,  we need to have a bird's eye view of the data. For now, let's have a look at how the bandgap values look like.

~~~
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10, 10))
plt.hist(band_gaps, bins=100)
plt.savefig('Histogram_PDF', bbox_inches='tight')
~~~
{: .python}

![](../fig/mp_bandgaps.png)


This plot shows that amost half of our structures are metals (zero bandgap). The bandgaps around 7 eV could be outliers, but we can deal with those in a later lecture. How about a line plot and scatter plot?

~~~
band_gaps_sorted=sorted(band_gaps)

# line plot
plt.figure(figsize=(10,10))
plt.plot(band_gaps_sorted)
plt.ylabel('')
plt.xlabel('')
plt.savefig('line plot', bbox_inches='tight')
~~~
{: .python}


# Part 3: Machine learnig
## Data Normalization 

Next, we put the data in a DataFrame and normalize it.

~~~
#!pip3 install -q sklearn pandas
import pandas as pd
from sklearn.preprocessing import StandardScaler

# We create a pandas DataFrame object

dataset_df = pd.DataFrame({"mean_atomic_numbers": mean_atomic_numbers,
                           "max_atomic_numbers": max_atomic_numbers,
                           "min_atomic_numbers": min_atomic_numbers,
                           "std_atomic_numbers": std_atomic_numbers,
                           "a_parameters": a_parameters,
                           "b_parameters": b_parameters,
                           "c_parameters": c_parameters,
                           "alpha_parameters": alpha_parameters,
                           "beta_parameters": beta_parameters,
                           "gamma_parameters": gamma_parameters,
                           "mean_distance_matrix": mean_distance_matrix,
                           "max_distance_matrix": max_distance_matrix,
                           "min_distance_matrix": min_distance_matrix,
                           "std_distance_matrix": std_distance_matrix
                           })

# We need to normalize the data using a scaler

# Define the scaler
scaler = StandardScaler().fit(dataset_df)

# Scale the train set
scaled_dataset_df = scaler.transform(dataset_df)
~~~
{: .python}


The last bit of code in this step is splitting the training and test set. Let's use the standard split: 80/20.
~~~
from sklearn.model_selection import train_test_split
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    scaled_dataset_df, band_gaps, test_size=.2, random_state=0)

~~~
{: .python}


## The machine learning training task. 

Now it's time to actually do machine learning. We will try two machine learning models: the random forests and the XGBOOST models. We will quantify the prediction performance using two measures: goodness of fit (R2) and the mean squared error (MSE).

~~~
!pip3 install -q xgboost

from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(n_estimators=400, max_depth=400, random_state=0)
regr.fit(X_train_scaled, y_train)
y_predicted = regr.predict(X_test_scaled)

print('RF MSE\t'+str(mean_squared_error(y_test, y_predicted))+'\n')
print('RF R2\t'+str(r2_score(y_test, y_predicted))+'\n')

xPlot=y_test
yPlot=y_predicted
plt.figure(figsize=(10,10))
plt.plot(xPlot,yPlot,'ro')
plt.plot(xPlot,xPlot)
plt.ylabel('RF')
plt.xlabel('DFT')
plt.savefig('RF_Correlation_Test', bbox_inches='tight')


regr = XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                    max_depth=400, alpha=10, n_estimators=400)
regr.fit(X_train_scaled, y_train)
y_predicted = regr.predict(X_test_scaled)

print('XGBOOST MSE\t'+str(mean_squared_error(y_test, y_predicted))+'\n')
print('XGBOOST R2\t'+str(r2_score(y_test, y_predicted))+'\n')


xPlot=y_test
yPlot=y_predicted
plt.figure(figsize=(10,10))
plt.plot(xPlot,yPlot,'ro')
plt.plot(xPlot,xPlot)
plt.ylabel('XGBOOST')
plt.xlabel('DFT')
plt.savefig('XGBOOST_Correlation_Test', bbox_inches='tight')

~~~
{: .python}


That doesn't look very impressive. The MLs are quite struggling with the metals. OK, let's redo this exercise and remove the metals.

~~~
import numpy as np
from pymatgen.io.cif import CifParser
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

mean_atomic_numbers = []
max_atomic_numbers = []
min_atomic_numbers = []
std_atomic_numbers = []
a_parameters = []
b_parameters = []
c_parameters = []
alpha_parameters = []
beta_parameters = []
gamma_parameters = []
mean_distance_matrix = []
max_distance_matrix = []
min_distance_matrix = []
std_distance_matrix = []
band_gaps = []

for doc in results:
    bg = doc.band_gap
    if bg > 1 and bg < 6:
        structure = doc.structure

        mean_atomic_numbers.append(np.mean(structure.atomic_numbers))
        max_atomic_numbers.append(np.max(structure.atomic_numbers))
        min_atomic_numbers.append(np.min(structure.atomic_numbers))
        std_atomic_numbers.append(np.std(structure.atomic_numbers))

        # Lattice parameters:
        a, b, c = structure.lattice.abc
        alpha, beta, gamma = structure.lattice.angles
        a_parameters.append(a)
        b_parameters.append(b)
        c_parameters.append(c)
        alpha_parameters.append(alpha)
        beta_parameters.append(beta)
        gamma_parameters.append(gamma)

        distance_matrix = structure.distance_matrix
        mean_distance_matrix.append(np.mean(distance_matrix))
        max_distance_matrix.append(np.max(distance_matrix))
        min_distance_matrix.append(np.min(distance_matrix))
        std_distance_matrix.append(np.std(distance_matrix))

        band_gaps.append(bg)

# How many do we have now?
print(len(band_gaps))

# Data visualization: let's have a look at our data.
plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(10, 10))
plt.hist(band_gaps, bins=100)
plt.savefig('Histogram_PDF_NoMetals', bbox_inches='tight')

band_gaps_sorted = sorted(band_gaps)

# Scatter plot
plt.figure(figsize=(10,10))
plt.plot(band_gaps_sorted)
plt.ylabel('')
plt.xlabel('')
plt.savefig('ScatterPlot_NoMetals', bbox_inches='tight')

# Next, we create a pandas DataFrame object
dataset_df = pd.DataFrame({
    "mean_atomic_numbers": mean_atomic_numbers,
    "max_atomic_numbers": max_atomic_numbers,
    "min_atomic_numbers": min_atomic_numbers,
    "std_atomic_numbers": std_atomic_numbers,
    "a_parameters": a_parameters,
    "b_parameters": b_parameters,
    "c_parameters": c_parameters,
    "alpha_parameters": alpha_parameters,
    "beta_parameters": beta_parameters,
    "gamma_parameters": gamma_parameters,
    "mean_distance_matrix": mean_distance_matrix,
    "max_distance_matrix": max_distance_matrix,
    "min_distance_matrix": min_distance_matrix,
    "std_distance_matrix": std_distance_matrix
})

# We need to normalize the data using a scaler
scaler = StandardScaler().fit(dataset_df)
scaled_dataset_df = scaler.transform(dataset_df)

# Then we do a 80/20 split of data: training set and test set
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    scaled_dataset_df, band_gaps, test_size=.2, random_state=None)

# Random Forest Regressor
regr = RandomForestRegressor(n_estimators=400, max_depth=400, random_state=0)
regr.fit(X_train_scaled, y_train)
y_predicted = regr.predict(X_test_scaled)

print('RF MSE\t'+str(mean_squared_error(y_test, y_predicted))+'\n')
print('RF R2\t'+str(r2_score(y_test, y_predicted))+'\n')

xPlot = y_test
yPlot = y_predicted
plt.figure(figsize=(10,10))
plt.plot(xPlot, yPlot, 'ro')
plt.plot(xPlot, xPlot)
plt.ylabel('RF')
plt.xlabel('DFT')
plt.savefig('RF_Correlation_Test', bbox_inches='tight')

# XGBoost Regressor
regr = XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                    max_depth=400, alpha=10, n_estimators=400)
regr.fit(X_train_scaled, y_train)
y_predicted = regr.predict(X_test_scaled)

print('XGBOOST MSE\t'+str(mean_squared_error(y_test, y_predicted))+'\n')
print('XGBOOST R2\t'+str(r2_score(y_test, y_predicted))+'\n')

xPlot = y_test
yPlot = y_predicted
plt.figure(figsize=(10,10))
plt.plot(xPlot, yPlot, 'ro')
plt.plot(xPlot, xPlot)
plt.ylabel('XGBOOST')
plt.xlabel('DFT')
plt.savefig('XGBOOST_Correlation_Test', bbox_inches='tight')
~~~
{: .python}


There is improvement in MSE, but not much in R2. However: our descriptors should be able to identify all bandgaps, not only non-metals. Therefore, more work needs to be done on the descriptors we developed here. A much improved set of descriptors are available in this notebook, with R2 ~ 0.51:

# Part 3: Deep learning for  materials science

## CHNet for Property Prediction

Crystal Hamiltonian Graph neural Network(CHNet) is pretrained on the GGA/GGA+U static and relaxation trajectories from Materials Project, a comprehensive dataset consisting of more than 1.5 Million structures from 146k compounds spanning the whole periodic table.

CHGNet highlights its ability to study electron interactions and charge distribution in atomistic modeling with near DFT accuracy. The charge inference is realized by regularizing the atom features with DFT magnetic moments, which carry rich information about both local ionic environments and charge distribution.

~~~
!pip install chgnet
import numpy as np
from pymatgen.core import Structure
from chgnet.model import CHGNet

# If the above line fails in Google Colab due to numpy version issue,
# please restart the runtime, and the problem will be solved
np.set_printoptions(precision=4, suppress=True)
~~~
{: .python}


~~~
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
# replace  YOUR_API_KEY with our own
with MPRester(api_key="YOUR_API_KEY") as mpr:
    # Retrieve the crystal structures for a list of materials
    docs = mpr.materials.summary.search(material_ids=["mp-18767"], fields=["material_id", "structure"])

    # Save the CIF files
    for doc in docs:
        material_id = doc.material_id
        structure = doc.structure
        filename = f"{material_id}.cif"
        
        # Convert the structure to CIF format and save
        cif = CifWriter(structure)
        cif.write_file(filename)
        print(f"Saved {filename}")
~~~
{: .python}


now read the saved cif file
~~~
structure = Structure.from_file(f"mp-18767.cif")
print(structure)
~~~
{: .python}

~~~
Full Formula (Li2 Mn2 O4)
Reduced Formula: LiMnO2
abc   :   2.806775   4.603051   5.701790
angles:  90.000005  90.000003  90.000000
pbc   :       True       True       True
Sites (8)
  #  SP           a         b         c
---  ----  --------  --------  --------
  0  Li    0.749999  0.75      0.880615
  1  Li    0.249999  0.249999  0.119384
  2  Mn    0.749997  0.75      0.362786
  3  Mn    0.249996  0.249999  0.637214
  4  O     0.75      0.249999  0.862143
  5  O     0.25      0.75      0.599339
  6  O     0.749999  0.25      0.40066
  7  O     0.25      0.75      0.137858
~~~
{: .output}


## Define Model

~~~
chgnet = CHGNet.load()
# Alternatively you can read your own model
# chgnet = CHGNet.from_file(model_path)
~~~
{: .python}

~~~
CHGNet v0.3.0 initialized with 412,525 parameters
CHGNet will run on cpu
~~~
{: .output}


## Predict energy, force, stress, magmom

~~~
prediction = chgnet.predict_structure(structure)
for key, unit in [
    ("energy", "eV/atom"),
    ("forces", "eV/A"),
    ("stress", "GPa"),
    ("magmom", "mu_B"),
]:
    print(f"CHGNet-predicted {key} ({unit}):\n{prediction[key[0]]}\n")
~~~
{: .python}

~~~
CHGNet-predicted energy (eV/atom):
-7.367691516876221

CHGNet-predicted forces (eV/A):
[[ 0.      0.      0.0238]
 [ 0.      0.     -0.0238]
 [ 0.      0.      0.0926]
 [-0.      0.     -0.0926]
 [ 0.     -0.     -0.0024]
 [-0.     -0.     -0.0131]
 [ 0.      0.      0.0131]
 [ 0.     -0.      0.0024]]

CHGNet-predicted stress (GPa):
[[-0.3042 -0.      0.    ]
 [-0.      0.223   0.    ]
 [-0.      0.     -0.1073]]

CHGNet-predicted magmom (mu_B):
[0.003  0.003  3.8694 3.8694 0.0441 0.0386 0.0386 0.0441]
~~~
{: .output}

## CHNet for Structure Optimization 

~~~
from chgnet.model import StructOptimizer

relaxer = StructOptimizer()
~~~
{: .python}

~~~
CHGNet v0.3.0 initialized with 412,525 parameters
CHGNet will run on cpu
~~~
{: .output}

~~~
# Perturb the structure
structure.perturb(0.1)

# Relax the perturbed structure
result = relaxer.relax(structure, verbose=True)
print("Relaxed structure:\n")
print(result["final_structure"])
~~~
{: .python}


~~~
   Step     Time          Energy          fmax
FIRE:    0 15:34:01      -58.351952        3.147362
FIRE:    1 15:34:02      -58.580257        2.058782
FIRE:    2 15:34:03      -58.789093        0.844543
FIRE:    3 15:34:03      -58.820564        0.960811
FIRE:    4 15:34:04      -58.826691        0.888024
FIRE:    5 15:34:05      -58.837467        0.760726
FIRE:    6 15:34:05      -58.850525        0.585002
FIRE:    7 15:34:06      -58.863132        0.404391
FIRE:    8 15:34:06      -58.872852        0.294677
FIRE:    9 15:34:07      -58.879227        0.337815
FIRE:   10 15:34:07      -58.883316        0.436549
FIRE:   11 15:34:08      -58.887302        0.524489
FIRE:   12 15:34:08      -58.893330        0.552167
FIRE:   13 15:34:09      -58.902809        0.469269
FIRE:   14 15:34:09      -58.914330        0.284212
FIRE:   15 15:34:09      -58.924210        0.152119
FIRE:   16 15:34:10      -58.928581        0.232751
FIRE:   17 15:34:10      -58.929062        0.369346
FIRE:   18 15:34:11      -58.929829        0.346120
FIRE:   19 15:34:11      -58.931175        0.301094
FIRE:   20 15:34:11      -58.932800        0.241701
FIRE:   21 15:34:12      -58.934364        0.169798
FIRE:   22 15:34:12      -58.935524        0.102815
FIRE:   23 15:34:13      -58.936115        0.099835
Relaxed structure:

Full Formula (Li2 Mn2 O4)
Reduced Formula: LiMnO2
abc   :   2.867063   4.636071   5.792108
angles:  90.081488  90.037062  90.005805
pbc   :       True       True       True
Sites (8)
  #  SP           a         b         c      magmom
---  ----  --------  --------  --------  ----------
  0  Li    0.739875  0.756814  0.878128  0.00351867
  1  Li    0.261777  0.244955  0.119803  0.00367078
  2  Mn    0.744893  0.747541  0.362541  3.86412
  3  Mn    0.252455  0.246545  0.634633  3.86501
  4  O     0.753446  0.246433  0.8599    0.0430375
  5  O     0.245003  0.749863  0.598456  0.0355659
  6  O     0.751379  0.249981  0.397878  0.0365092
  7  O     0.244749  0.752093  0.136739  0.0435092
~~~
{: .output}


## CHNet for Molecular Dynamics

~~~
from chgnet.model.dynamics import MolecularDynamics

md = MolecularDynamics(
    atoms=structure,
    model=chgnet,
    ensemble="nvt",
    temperature=1000,  # in k
    timestep=2,  # in fs
    trajectory="md_out.traj",
    logfile="md_out.log",
    loginterval=100,
)
md.run(50)  # run a 0.1 ps MD simulation
~~~
{: .python}


~~~
CHGNet will run on cpu
NVT-Berendsen-MD created
~~~
{: .output}

## Magmom Visualization

~~~
supercell = structure.make_supercell([2, 2, 2], in_place=False)print(supercell.composition)
~~~
{: .python}

~~~
Li16 Mn16 O32
~~~
{: .output}

~~~
import random
n_Li = int(supercell.composition["Li+"])
remove_ids = random.sample(list(range(n_Li)), n_Li // 2)

supercell.remove_sites(remove_ids)
print(supercell.composition)
~~~
{: .python}


~~~
Li16 Mn16 O32
~~~
{: .output}

~~~
result = relaxer.relax(supercell)
~~~
{: .python}

~~~
      Step     Time          Energy          fmax
FIRE:    0 15:35:54     -466.815552        3.147368
FIRE:    1 15:35:57     -468.534790        2.103104
FIRE:    2 15:35:59     -469.953186        0.993437
FIRE:    3 15:36:01     -470.422852        0.717387
FIRE:    4 15:36:03     -470.223114        1.079820
FIRE:    5 15:36:05     -470.288544        0.999953
FIRE:    6 15:36:07     -470.402954        0.857044
FIRE:    7 15:36:09     -470.537384        0.644330
FIRE:    8 15:36:12     -470.660156        0.439064
FIRE:    9 15:36:17     -470.739319        0.427616
FIRE:   10 15:36:20     -470.771942        0.437620
FIRE:   11 15:36:22     -470.770691        0.480304
FIRE:   12 15:36:28     -470.773743        0.472451
FIRE:   13 15:36:32     -470.779755        0.457100
FIRE:   14 15:36:35     -470.788239        0.455132
FIRE:   15 15:36:37     -470.798676        0.452816
FIRE:   16 15:36:38     -470.810394        0.450266
FIRE:   17 15:36:40     -470.822815        0.447701
FIRE:   18 15:36:42     -470.835388        0.445307
FIRE:   19 15:36:44     -470.848816        0.442580
FIRE:   20 15:36:46     -470.862579        0.439440
FIRE:   21 15:36:48     -470.876221        0.437258
FIRE:   22 15:36:49     -470.889221        0.435443
FIRE:   23 15:36:52     -470.902527        0.435156
FIRE:   24 15:36:53     -470.917480        0.435341
FIRE:   25 15:36:55     -470.935791        0.434604
FIRE:   26 15:36:57     -470.958984        0.430960
FIRE:   27 15:36:59     -470.987274        0.424823
FIRE:   28 15:37:01     -471.017914        0.417879
FIRE:   29 15:37:03     -471.045593        0.409992
FIRE:   30 15:37:05     -471.068451        0.398345
FIRE:   31 15:37:06     -471.091248        0.383798
FIRE:   32 15:37:08     -471.120148        0.368756
FIRE:   33 15:37:09     -471.154846        0.353476
FIRE:   34 15:37:11     -471.188080        0.336356
FIRE:   35 15:37:14     -471.224365        0.317093
FIRE:   36 15:37:15     -471.270081        0.294736
FIRE:   37 15:37:17     -471.308777        0.269416
FIRE:   38 15:37:18     -471.342621        0.235037
FIRE:   39 15:37:20     -471.382996        0.195981
FIRE:   40 15:37:21     -471.420715        0.155945
FIRE:   41 15:37:23     -471.468750        0.111853
FIRE:   42 15:37:25     -471.499359        0.096404
~~~
{: .output}


~~~
import pandas as pd
df_magmom = pd.DataFrame({"Unrelaxed": chgnet.predict_structure(supercell)["m"]})
df_magmom["CHGNet relaxed"] = result["final_structure"].site_properties["magmom"]
~~~
{: .python}

~~~
fig = df_magmom.hist(
    nbins=200,
    sharex=True,
    sharey=True,
    backend="plotly",
    barmode="overlay",
    layout={"title": "Magmom distribution"},
    opacity=0.7,
    range_x=[3, 4],
    template="plotly_white",
)
fig.layout.legend.update(title="", x=1, y=1, xanchor="right", yanchor="top")
fig.layout.xaxis.title = "Magnetic moment"
fig.show()
~~~
{: .python}



## MP Formation energy

The pre-trained models are based on the Materials Project  dataset. There are two models available - MEGNet and M3GNet.

We create the structure first. This is based on the relaxed structure obtained from the Materials Project. Alternatively, one can use the Materials Project API to obtain the structure.


~~~
from __future__ import annotations
import warnings
import torch
from pymatgen.core import Lattice, Structure
import matgl
# To suppress warnings for clearer output
warnings.simplefilter("ignore")

struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

~~~
{: .python}




### Matgl for Prediction


