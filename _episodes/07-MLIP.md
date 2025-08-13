---
title: MLIP
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

## Materials Graph Library

Certainly! Here's a rewritten version of your text:

***

## Materials Graph Library

MatGL (Materials Graph Library) is a specialized graph deep learning library designed for materials science. Since mathematical graphs naturally represent collections of atoms, graph deep learning models are highly effective as surrogate models for predicting material properties, consistently delivering outstanding performance.

Built on top of the Deep Graph Library (DGL) and PyTorch, MatGL incorporates specific adaptations tailored to materials science applications. Its purpose is to provide an extensible platform for developing and sharing graph deep learning models for materials, such as the MatErials 3-body Graph Network (M3GNet) and its predecessor, MEGNet.

This project is a collaborative effort between the Materials Virtual Lab and Intel Labs, with contributions from Santiago Miret, Marcel Nassar, and Carmelo Gonzales.

MatGL is a key component of the MatML ecosystem, which also includes the maml (Materials Machine Learning) package, the MatPES (Materials Potential Energy Surface) dataset, and the MatCalc (Materials Calculator) tool.

## MatErials Graph Network (MEGNet)

MEGNet is a graph network implementation inspired by DeepMind’s approach, tailored for machine learning in materials science. It has proven highly effective, achieving low prediction errors across a wide range of properties for both molecules and crystals. Recent updates include advancements in multi-fidelity materials property modeling. Figure 1 illustrates the sequential update process of the graph network, where bonds, atoms, and global state attributes are iteratively updated by exchanging information, ultimately producing an updated output graph.

## M3GNet

The Materials 3-body Graph Network (M3GNet) is an enhanced graph neural network architecture that extends MEGNet by incorporating three-body interactions. A notable addition in M3GNet is the inclusion of atomic coordinates and the 3×3 lattice matrix for crystals, which are essential for calculating tensorial quantities like forces and stresses through auto-differentiation. 

M3GNet serves as a versatile framework with several key applications:

- **Interatomic potential development:** When trained on the same datasets, M3GNet matches the performance of state-of-the-art machine learning interatomic potentials (MLIPs). A major advantage of its graph-based representation is its scalability across diverse chemical spaces. Notably, M3GNet has enabled the creation of a universal interatomic potential that covers the entire periodic table, trained using relaxation data from the Materials Project.
  
- **Surrogate models for property prediction:** Building on the success of MEGNet, M3GNet can also be employed to build surrogate models for predicting material properties, often achieving accuracy equal to or better than other leading ML models.

For detailed benchmark results, please see the relevant publications in the References section.

MatGL offers a reimplementation of M3GNet using Deep Graph Library (DGL) and PyTorch. Compared to the original TensorFlow version, this implementation introduces several notable improvements:

- A more intuitive API and class structure powered by DGL.
- Support for multi-GPU training through PyTorch Lightning.


