# Lesson: Setting Up Python Environments on Ubuntu

---

## Objective

By the end of this lesson, you will know how to set up and use Python environments with **Conda** and **virtual environments (venv or virtualenv)**. This ensures you can manage dependencies and isolate projects effectively.

---

### Part 1: Using Conda to Set Up a Virtual Environment

#### Step 1: Creating the Environment
- Open the terminal.
- Type the following command to create a Conda environment named **torch** with Python 3.9:

  ```bash
  conda create -n torch python=3.9
  ```

- Replace `3.9` with your required Python version if necessary.

**Key Points:**
- `torch` is the environment name.
- `python=3.9` specifies the Python version.

#### Step 2: Activating the Environment
- Activate the environment with:

  ```bash
  conda activate torch
  ```

**Key Points:**
- The terminal prompt will change, showing the name of the active environment (torch).

#### Step 3: Installing Packages
- Install packages using `conda` or `pip`:

  ```bash
  conda install <package_name>
  pip install <package_name>
  ```

**Key Points:**
- Use Conda for faster and precompiled package installations.
- Use Pip when a package is unavailable in Conda.

#### Step 4: Deactivating the Environment
- Exit the environment with:

  ```bash
  conda deactivate
  ```

**Key Points:**
- Deactivating ensures you return to your system's default Python environment.

---

### Part 2: Setting Up a Virtual Environment with venv or virtualenv

#### Method 1: Using `venv` (Built-in Python Module)

1. **Create the Environment**
   - Navigate to your project directory:
     
     ```bash
     cd Documents/my_project
     ```
   - Create a virtual environment named **torch**:

     ```bash
     python3 -m venv torch
     ```

2. **Activate the Environment**
   - Run the following command to activate:

     ```bash
     source torch/bin/activate
     ```

   **Key Points:**
   - After activation, your terminal prompt will show `(torch)` to indicate the active environment.

3. **Install Required Packages**
   - Use pip to install dependencies:

     ```bash
     pip install <package_name>
     ```

4. **Deactivate the Environment**
   - Exit the environment with:

     ```bash
     deactivate
     ```

---

#### Method 2: Using `virtualenv` (Optional)

1. **Install Virtualenv**
   - Update your system's package lists and install virtualenv:

     ```bash
     sudo apt update
     sudo apt install python3-virtualenv
     ```

2. **Create the Environment**
   - Navigate to your project folder and create the environment:

     ```bash
     cd Documents/my_project
     virtualenv torch
     ```

3. **Activate the Environment**
   - Run the activation command:

     ```bash
     source torch/bin/activate
     ```

4. **Install Required Packages**
   - Use pip to install any necessary packages:

     ```bash
     pip install <package_name>
     ```

5. **Deactivate the Environment**
   - Exit the environment with:

     ```bash
     deactivate
     ```

---

### Exercises for Learners

1. Create a Conda environment named **torch_env** with Python 3.12. Install the `numpy` package.
2. Set up a virtual environment using `venv` named **torch_venv**. Activate it and install the `pandas` package.
3. Deactivate both environments after installation.

---

### Recap
- **Conda** is best for managing environments with both Python and non-Python dependencies.
- **venv** and **virtualenv** are ideal for simple Python projects.