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
- "Using `venv` for creating environments."
- "Activating and deactivating virtual environments."
---

# Virtual Environments in Python

A **virtual environment** is a self-contained directory that contains a Python installation for a particular version of Python, plus several additional packages. Using virtual environments, you can ensure that each project has its own dependencies, regardless of what dependencies every other project has.

### Why Use Virtual Environments?
- **Isolation**: Each project can have its own dependencies without affecting other projects.
- **Avoid conflicts**: Different projects can use different versions of libraries.
- **Simplify deployment**: Virtual environments can make it easier to move projects between different machines or systems.

### How to Create a Virtual Environment

To create a virtual environment, you can use the built-in `venv` module in Python. Below is how you can do it.

**1. Create the Virtual Environment**

In the terminal, navigate to the project directory and run the following command to create a virtual environment:

~~~
python3 -m venv myenv
~~~  
{: .bash}

This will create a new directory called `myenv` containing the virtual environment.

**2. Activate the Virtual Environment**

- **On Windows**: 
  ~~~
  .\myenv\Scripts\activate
  ~~~
  {: .bash}
  
- **On macOS and Linux**:
  ~~~
  source myenv/bin/activate
  ~~~
  {: .bash}

Once activated, the name of the environment (e.g., `myenv`) will appear in the terminal prompt, indicating that the virtual environment is active.

### Installing Packages

Once the virtual environment is activated, you can use `pip` to install the packages you need for your project. For example:

~~~
pip install numpy
~~~  
{: .bash}

### Deactivating the Virtual Environment

To deactivate the virtual environment, simply run:

~~~
deactivate
~~~  
{: .bash}

This will return you to the global Python environment.

---

## Exercise: Create and Use a Virtual Environment

1. Create a new virtual environment for a new project named `myproject`.
2. Install the package `requests` in the virtual environment.
3. Write a simple Python script that uses `requests` to fetch a webpage (use `requests.get()` to get the page).
4. Deactivate the virtual environment after running the script.

**Solution**:
Here is how you can create and activate the virtual environment, install `requests`, and use it in a Python script:

1. **Create the virtual environment**:

~~~
python3 -m venv myproject
~~~  
{: .bash}

2. **Activate the environment**:

- **On Windows**:  
  ~~~
  .\myproject\Scripts\activate
  ~~~
  {: .bash}

- **On macOS/Linux**:  
  ~~~
  source myproject/bin/activate
  ~~~
  {: .bash}

3. **Install `requests`**:

~~~
pip install requests
~~~  
{: .bash}

4. **Python script using `requests`**:

Create a file named `fetch_page.py`:

~~~python
import requests

response = requests.get('https://www.example.com')
print(response.text)
~~~  
{: .python}

5. **Deactivate the environment**:

~~~
deactivate
~~~  
{: .bash}
