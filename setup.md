---
layout: page
title: Setup
permalink: /setup/
---

# Python environment Managment

Python environment management with conda involves creating, managing, and switching between different Python environments, each with its own set of packages and dependencies. Conda is a package, dependency, and environment management tool that is widely used in the scientific community, especially on the Windows platform where the installation of binary extensions can be difficult

## What is a Conda environment

A Conda environment is a directory that contains a specific collection of Conda packages that you have installed. For example, you may be working on a research project that requires NumPy 1.18 and its dependencies, while another environment associated with an finished project has NumPy 1.12 (perhaps because version 1.12 was the most current version of NumPy at the time the project finished). If you change one environment, your other environments are not affected. You can easily activate or deactivate environments, which is how you switch between them.

> Avoid installing packages into your base Conda environment

Conda has a default environment called base that include a Python installation and some core system libraries and dependencies of Conda. It is a “best practice” to avoid installing additional packages into your base software environment. Additional packages needed for a new project should always be installed into a newly created Conda environment.


## Creating Environments on Linux/MacOS with conda (Recommended)


Conda environments behave similarly to global environments - installed packages are available to all projects using that environment. It allows you to create environments that isolate each project, thereby preventing dependency conflicts between projects. You can create a new environment with a specific version of Python and multiple packages using the following command:

~~~
admin@MacBook~ $ conda create -n <env_name> python=<version#> 
~~~
{: bash}

For instance, to create a new conda environment called `pygmt` with Python 3.12

~~~
admin@MacBook~ $ conda create --name ml-climate python=3.12
~~~
{: .bash}

To activate the environment:

~~~
admin@MacBook~ $ conda activate ml-climate
~~~
{: .bash}



In order to make your results more reproducible and to make it easier for research colleagues to recreate your Conda environments on their machines it is a “best practice” to always explicitly specify the version number for each package that you install into an environment. If you are not sure exactly which version of a package you want to use, then you can use search to see what versions are available using the conda search command.


~~~
admin@MacBook~ $ conda search $PACKAGE_NAME
~~~
{: .bash}

So, for example, if you wanted to see which versions of Scikit-learn, a popular Python library for machine learning, were available, you would run the following.


~~~
admin@MacBook~ $ conda search xarray
~~~
{: .bash}

In order to create a new environment you use the conda create command as follows.


~~~
admin@MacBook~ $ conda create --name ml-flow \
 scikit-learn \
 geopandas \
 cartopy \
 torch \
 xarray

~~~
{: .bash}

### Creating an environment from a YAML file

Now let’s do the reverse operation and create an environment from a yaml file. You will find these files often in GitHub repositories, so it is handy to know how to use them. Let’s open a text editor and make some changes to our myenv.yaml file, so that it looks like this:

~~~
name: ml-flow
channels:
  - defaults
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/free
dependencies:
  - numpy
  - python=3.12
  - xarray
  - geopandas
  - pysal
  - gmt
  - gdal
  - dask
  - hdf5
  - hvplot
~~~
{: .yaml}

### Deactivating environments

### Exporting environments

The next command we will cover in this workshop lets us export the configuration of an environment to a file, so that we can share it with others. Instead of bundling the packages themselves, `conda` exports a list of the package names and their versions, as we have them on our system. In addition to package details, the file contains also the list of all channels we defined in our configuration, both globally and environment-specific. Finally, the file is written in `YAML`, a human-readable text format that we can inspect manually and edit if necessary. Let’s export the `pygmt` environment to a file:

~~~
admin@MacBook~ $ conda env export --no-builds --file ml-flow.yaml
~~~
{: .bash}

There are some options to unpack in this command. First, we do not use conda but the conda env subcommand, which is a more advanced script to manage environments. We also pass a --no-builds option, which tells conda to specify only the package names and versions. By default, conda would have exported the build information for each package, which you can think of as a very precise version that is sometimes specific to the operative system. While this is great for reproducibility, it is very likely that you will end up not being able to share your environment across different systems.

### Install Required Packages

~~~
admin@MacBook~ $ conda install conda-forge::<package_name>
~~~
{: .bash}

or 

~~~
admin@MacBook~ $ pip install <package_name>
~~~
{: .bash}



### Removing environments

Finally, let’s see how we remove environments. Removing environments is useful when you make mistakes and environments become unusable, or just because you finished a project and you need to clear some disk space. The command to remove an environment is the following:

~~~
admin@MacBook~ $ conda env remove --name $envname
~~~
{: .bash}



## Creating Environments with venv Module 

### 1) Open a Terminal

- Launch the terminal by pressing `Ctrl + Alt + T` or searching for `Terminal` in your application launcher.

### 2) Navigate to the Desired Directory

- Use the `cd` command to move to the directory where you want to create the virtual environment. For example:

~~~
admin@MacBook~ $ cd Documents/ml-climate
~~~
{: .bash}

### 3) Create the Virtual Environment

- Execute the `python3 -m venv` command with your chosen environment name:

~~~
admin@MacBook~ $ python3 -m venv ml-climate
~~~
{: .bash}

- This will create a directory named `venv` for your virtual environment in the specified location.

### 4) Activate the Virtual Environment

- Activate the virtual environment by running the following command, adjusting for your environment name

~~~
admin@MacBook~ $ source venv/bin/activate
~~~
{: .bash}

### 5) Install Required Packages

- With the environment activated, use pip to install the necessary packages:

~~~
admin@MacBook~ $ pip install <package_name>
~~~
{: .bash}

### 6) Deactivate the Environment (Optional)

- To exit the virtual environment, type:

~~~
admin@MacBook~ $ deactivate
~~~
{: .bash}

- This command deactivates the virtual environment and returns you to the default Python environment.

## Method 2: Using virtualenv (Optional)

This alternative method might require first installing virtualenv.

### Update Package Lists

- Update your package lists with:

~~~
admin@MacBook~ $ sudo apt update 
~~~
{: .bash}

### Install virtualenv

- Install virtualenv using:

~~~
admin@MacBook~ $ sudo apt install python3-venv
~~~

### Create the Virtual Environment

- Use the `virtualenv` command with your chosen environment name:

~~~
admin@MacBook~ $ virtualenv  venv
~~~
{: .bash}

- This will create a directory named `venv` for your virtual environment in the specified location.

### Activate the Virtual Environment

- Activate the virtual environment by running the following command, adjusting for your environment name:

~~~
admin@MacBook~ $ source venv/bin/activate
~~~
{: .bash}

### Install Required Packages

- With the environment activated, use pip to install the necessary packages:

~~~
admin@MacBook~ $ pip install <package_name>
~~~
{: .bash}


### Deactivate the Environment (Optional)

When you're done working in your virtual environment and want to return to your system's default Python environment, you can exit the virtual environment. To do this, simply type:

~~~
admin@MacBook~ $ deactivate
~~~
{: .bash}

This command deactivates the virtual environment, restoring the original environment settings and paths. After deactivation, your terminal will return to using the system-wide Python installation and its associated packages.


## Creating Environments on Windows with conda (Recommended)


