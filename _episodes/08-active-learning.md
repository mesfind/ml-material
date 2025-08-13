---
title:  Active Learning 
teaching: 1
exercises: 0
questions:
- ""
objectives:
- ""
---

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>


# Active Learning for materials


Active learning is a data acquisition approach where the algorithm actively collaborates with a user to gather additional labeled data, improving its accuracy over time. Initially, the algorithm is trained on a small set of labeled examples and then identifies which new data points would be most beneficial to label. It queries the user to label these uncertain or informative samples, using the newly labeled data to enhance its performance. This cycle continues until the model reaches a desired level of accuracy.

**Example:** Consider training a machine learning model to recognize handwritten digits. The process begins with a limited set of labeled digits. The algorithm then selects digits it finds uncertain and asks the user to label them. By incorporating these newly labeled samples, the model refines its predictions, repeating this interactive labeling until it can reliably identify most handwritten digits.

## Derivative structure enumeration

~~~
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation
from pymatgen.io.vasp.sets import batch_write_input, MPRelaxSet
structure = Structure.from_file("EntryWithCollCode418490.cif")
print(structure)
# loop over all sites in the structure
for i, site in enumerate(structure):
    # change the occupancy of Li+ disordered sites to 0.5
    if not site.is_ordered:
        structure[i] = {"Li+": 0.5}
print("The composition after adjustments is %s." % structure.composition.reduced_formula)
analyzer = SpacegroupAnalyzer(structure)
prim_cell = analyzer.find_primitive()
print(prim_cell)
enum = EnumerateStructureTransformation()
enumerated = enum.apply_transformation(prim_cell, 100)  # return no more than 100 structures
structures = [d["structure"] for d in enumerated]  
print("%d structures returned." % len(structures))
~~~
{: .python}

~~~
Full Formula (Li26.88 P4 S20 Cl4)
Reduced Formula: Li26.88P4S20Cl4
abc   :   9.859000   9.859000   9.859000
angles:  90.000000  90.000000  90.000000
pbc   :       True       True       True
Sites (76)
  #  SP              a        b        c
---  --------  -------  -------  -------
  0  Li+:0.56  0.3148   0.982    0.3148
  1  Li+:0.56  0.982    0.6852   0.6852
  2  Li+:0.56  0.6852   0.3148   0.018
  3  Li+:0.56  0.3148   0.6852   0.018
  4  Li+:0.56  0.982    0.3148   0.3148
  5  Li+:0.56  0.6852   0.982    0.6852
  6  Li+:0.56  0.3148   0.018    0.6852
  7  Li+:0.56  0.6852   0.6852   0.982
  8  Li+:0.56  0.3148   0.3148   0.982
  9  Li+:0.56  0.6852   0.018    0.3148
 10  Li+:0.56  0.018    0.6852   0.3148
 11  Li+:0.56  0.018    0.3148   0.6852
 12  Li+:0.56  0.3148   0.482    0.8148
 13  Li+:0.56  0.982    0.1852   0.1852
 14  Li+:0.56  0.6852   0.8148   0.518
 15  Li+:0.56  0.3148   0.1852   0.518
 16  Li+:0.56  0.982    0.8148   0.8148
 17  Li+:0.56  0.6852   0.482    0.1852
 18  Li+:0.56  0.3148   0.518    0.1852
 19  Li+:0.56  0.6852   0.1852   0.482
 20  Li+:0.56  0.3148   0.8148   0.482
 21  Li+:0.56  0.6852   0.518    0.8148
 22  Li+:0.56  0.018    0.1852   0.8148
 23  Li+:0.56  0.018    0.8148   0.1852
 24  Li+:0.56  0.8148   0.982    0.8148
 25  Li+:0.56  0.482    0.6852   0.1852
 26  Li+:0.56  0.1852   0.3148   0.518
 27  Li+:0.56  0.8148   0.6852   0.518
 28  Li+:0.56  0.482    0.3148   0.8148
 29  Li+:0.56  0.1852   0.982    0.1852
 30  Li+:0.56  0.8148   0.018    0.1852
 31  Li+:0.56  0.1852   0.6852   0.482
 32  Li+:0.56  0.8148   0.3148   0.482
 33  Li+:0.56  0.1852   0.018    0.8148
 34  Li+:0.56  0.518    0.6852   0.8148
 35  Li+:0.56  0.518    0.3148   0.1852
 36  Li+:0.56  0.8148   0.482    0.3148
 37  Li+:0.56  0.482    0.1852   0.6852
 38  Li+:0.56  0.1852   0.8148   0.018
 39  Li+:0.56  0.8148   0.1852   0.018
 40  Li+:0.56  0.482    0.8148   0.3148
 41  Li+:0.56  0.1852   0.482    0.6852
 42  Li+:0.56  0.8148   0.518    0.6852
 43  Li+:0.56  0.1852   0.1852   0.982
 44  Li+:0.56  0.8148   0.8148   0.982
 45  Li+:0.56  0.1852   0.518    0.3148
 46  Li+:0.56  0.518    0.1852   0.3148
 47  Li+:0.56  0.518    0.8148   0.6852
 48  P5+       0.5      0        0
 49  P5+       0        0        0.5
 50  P5+       0        0.5      0
 51  P5+       0.5      0.5      0.5
 52  S2-       0.25     0.75     0.25
 53  S2-       0.75     0.75     0.75
 54  S2-       0.75     0.25     0.25
 55  S2-       0.25     0.25     0.75
 56  S2-       0.38053  0.11947  0.11947
 57  S2-       0.11947  0.88053  0.61947
 58  S2-       0.88053  0.38053  0.88053
 59  S2-       0.38053  0.88053  0.88053
 60  S2-       0.11947  0.38053  0.11947
 61  S2-       0.88053  0.11947  0.61947
 62  S2-       0.11947  0.11947  0.38053
 63  S2-       0.88053  0.61947  0.11947
 64  S2-       0.11947  0.61947  0.88053
 65  S2-       0.88053  0.88053  0.38053
 66  S2-       0.61947  0.11947  0.88053
 67  S2-       0.61947  0.88053  0.11947
 68  S2-       0.38053  0.61947  0.61947
 69  S2-       0.38053  0.38053  0.38053
 70  S2-       0.61947  0.61947  0.38053
 71  S2-       0.61947  0.38053  0.61947
 72  Cl-       0        0        0
 73  Cl-       0        0.5      0.5
 74  Cl-       0.5      0        0.5
 75  Cl-       0.5      0.5      0
The composition after adjustments is Li6PS5Cl.
Full Formula (Li6 P1 S5 Cl1)
Reduced Formula: Li6PS5Cl
abc   :   6.971366   6.971366   6.971366
angles:  60.000000  60.000000  60.000000
pbc   :       True       True       True
Sites (19)
  #  SP             a        b        c
---  -------  -------  -------  -------
  0  Li+:0.5  0.018    0.3524   0.018
  1  Li+:0.5  0.018    0.018    0.6116
  2  Li+:0.5  0.018    0.6116   0.3524
  3  Li+:0.5  0.018    0.3524   0.6116
  4  Li+:0.5  0.018    0.018    0.3524
  5  Li+:0.5  0.018    0.6116   0.018
  6  Li+:0.5  0.3524   0.018    0.6116
  7  Li+:0.5  0.6116   0.018    0.018
  8  Li+:0.5  0.3524   0.018    0.018
  9  Li+:0.5  0.6116   0.018    0.3524
 10  Li+:0.5  0.6116   0.3524   0.018
 11  Li+:0.5  0.3524   0.6116   0.018
 12  P5+      0.5      0.5      0.5
 13  S2-      0.25     0.25     0.25
 14  S2-      0.61947  0.61947  0.14159
 15  S2-      0.61947  0.14159  0.61947
 16  S2-      0.61947  0.61947  0.61947
 17  S2-      0.14159  0.61947  0.61947
 18  Cl-      0        0        0
48 structures returned.
~~~
{: .output}
