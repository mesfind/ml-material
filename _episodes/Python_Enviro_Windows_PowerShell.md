# Setting Up a Python Environment on Windows PowerShell

This guide provides step-by-step instructions on setting up a Python environment for machine learning training in Windows PowerShell. A virtual environment isolates project-specific packages and dependencies, ensuring a clean and organized workflow.

---

Prerequisites:

- Windows Operating System: This guide assumes you're using a Windows machine.
  
- Administrator Privileges (Optional): Certain commands might require administrator rights to create directories and install software. Right-click on "Windows PowerShell" and select "Run as administrator" if necessary.


## Steps:

### 1) Open Windows PowerShell

- Open the Start menu and search for `Windows PowerShell`.

- Right-click on `Windows PowerShell` and select `Open` (or run as administrator if needed).

### 2) Navigate to Drive D: (replace with your desired drive)

- Use the `Set-Location` cmdlet to change to the desired drive

`Set-Location -Path D:` or cd `D:`

### 3) Create a New Directory

- Use the `New-Item` cmdlet to create a directory named ml_train (or any preferred name):

`New-Item -ItemType Directory -Path ml_train`  or `mkdir ml_train`

### 4) Navigate to the New Directory

- Change the current directory to `ml_train` using `Set-Location`:

`Set-Location -Path ./ml_train`

### 5) Create a Virtual Environment

- Create a virtual environment:

`python -m venv ml_env`

### 6) Activate the Virtual Environment

`Set-ExecutionPolicy Bypass -Scope Process -Force`

`.\ml_env\Scripts\Activate.ps1`

### 7) Upgrade pip (optional)

- Upgrade pip (the Python package installer) to the latest version:

`python.exe -m pip install --upgrade pip`

### 8) Install Jupyter Notebook

`pip install notebook`

### 9) Verify Installation

To verify that Jupyter Notebook is installed correctly, you can start it:

`jupyter notebook`

- This will open Jupyter Notebook in your default web browser. You can then start creating and running notebooks.

### 10) Deactivate the Environment (Optional)

When you're finished, deactivate the environment by typing:

`deactivate`

- This exits the virtual environment and returns you to your system's default Python environment.

---

🥇Congratulations! You have successfully set up a Python environment with Jupyter Notebook on Windows using PowerShell.








