# Setting Up a Python virtual environment on Ubuntu 
---

## Method 1: Using venv Module (Recommended)

### 1) Open a Terminal

- Press `Ctrl + Alt + T` or search for `Terminal` in your application launcher.

### 2) Navigate to the Desired Directory

- Use the `cd` command to navigate to the location where you want to create the virtual environment. For example:

`cd Documents/my_project`

### 3) Create the Virtual Environment

- Use the `python3 -m venv` command followed by your desired environment name:

`python3 -m venv my_env`

- This creates a virtual environment directory named `my_env` in the current location.

### 4) Activate the Virtual Environment

- Activate the environment by running the following command (replace my_env accordingly):

`source my_env/bin/activate`

### 5) Install Required Packages

- Once the environment is activated, use pip to install the packages you need for your project:

`pip install <package_name>`

### 6) Deactivate the Environment (Optional)

When you're finished, deactivate the environment by typing:

`deactivate`

- This exits the virtual environment and returns you to your system's default Python environment.

---

## Method 2: Using virtualenv (Optional)

This method might require installing virtualenv first:

### 1) Update package lists

`sudo apt update`  

### 2) Install virtualenv

`sudo apt install python3-venv`

### 3) Create the Virtual Environment

- Use the `virtualenv` command followed by your desired environment name:

`virtualenv my_env`

- This creates a virtual environment directory named `my_env` in the current location.

### 4) Activate the Virtual Environment

- Activate the environment by running the following command (replace my_env accordingly):

`source my_env/bin/activate`

### 5) Install Required Packages

- Once the environment is activated, use pip to install the packages you need for your project:

`pip install <package_name>`

### 6) Deactivate the Environment (Optional)

When you're finished, deactivate the environment by typing:

`deactivate`

- This exits the virtual environment and returns you to your system's default Python environment.

## Method 3: Python virtual environment using the conda package manager

### 1. Create the Environment

- Open a terminal window. Use the conda create command followed by the desired environment name and the Python version you want.

- Replace "my_env" with your preferred name and "3.9" with the Python version

`conda create -n my_env python=3.9`

- This command creates a new environment named my_env with Python version 3.9 (adjust the version as needed).
  
- The -n flag specifies the environment name.

### 2. Activate the Environment

Once created, activate the environment using the following command (adjust the environment name if needed):

`conda activate my_env`

### 3. Deactivate the Environment

When finished, deactivate the environment by typing:

`conda deactivate`

---

If you dont have conda installed on your system.

Install Anaconda on Windows OS:

- https://www.anaconda.com/download

Install Anaconda on Ubuntu OS. 

- https://yonsci.github.io/yon_academic//portfolio/portfolio-2/

---
