# Setting Up a Python Environment on Windows Command Prompt
---

This guide provides step-by-step instructions on setting up a **Python environment** for machine learning training in **Windows Command Prompt**.

A virtual environment isolates project-specific packages and dependencies, ensuring a clean and organized workflow.

Prerequisites:

- Windows Operating System: This guide assumes you're using a Windows machine.

- Administrator Privileges (Optional): Certain commands might require administrator rights to create directories and install software. Right-click on "Command Prompt" and select "Run as administrator" if necessary.

## Steps 

### 1. Open Command Prompt

- Open the Command Prompt on your Windows machine. You can do this by pressing `Win + R`, typing `cmd,` and pressing `Enter`.

### 2. Navigate to Drive D: (replace with your desired drive)
- Switch to drive D: by typing the following command and pressing Enter:

`cd D:`

### 3. Create a New Directory
- Create a new directory named `ml_train` or any name you want by typing in the command line and pressing `Enter`:

`mkdir ml_train`

### 4. Navigate to the New Directory
- Change the current directory to `ml_train` by typing the following command and pressing `Enter`:

`cd ml_train`

### 5. Create a Virtual Environment
- Create a new virtual environment named `ml_env` by typing the following command and pressing `Enter`:

`python -m venv ml_env`

### 6. Activate the Virtual Environment
- Activate the virtual environment by typing the following command and pressing `Enter`:

`ml_env\Scripts\activate`


### 7. Upgrade pip (optional)
- Upgrade pip (the Python package installer) to the latest version by typing the following command and pressing `Enter`:

`python.exe -m pip install --upgrade pip`

### 8. Install Jupyter Notebook
- Install Jupyter Notebook by typing the following command and pressing `Enter`:

`pip install notebook`

### 9. Verify Installation
- To verify that Jupyter Notebook is installed correctly, you can start it by typing the following command and pressing `Enter`:

`jupyter notebook`

This will open Jupyter Notebook in your default web browser, and you can start creating and running Jupyter notebooks.

### 10) Deactivate the Environment (Optional)

When you're finished, deactivate the environment by typing:

`deactivate`

- This exits the virtual environment and returns you to your system's default Python environment.

---

🥇 Congratulations! You have successfully set up a Python environment with Jupyter Notebook on Windows.







