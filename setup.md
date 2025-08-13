---
layout: page
title: Setup
permalink: /setup/
---


## Software Setup

### Python

[Python](http://python.org) is a popular language for research computing and general-purpose programming. Installing all research packages individually can be challenging, so we recommend [Anaconda](https://www.continuum.io/anaconda), an all-in-one installer.

- Please install Python version 3.x (e.g., 3.9 is fine).
- We will use the IPython notebook, a programming environment running in a web browser. Use an up-to-date browser such as Chrome, Safari, or Firefox. Older browsers, like Internet Explorer 9 or earlier, are not supported ([details here](http://ipython.org/ipython-doc/2/install/install.html#browser-compatibility)).

#### Windows

[Video Tutorial](https://www.youtube.com/watch?v=xxQ0mzZ8UvA)

1. Open [http://continuum.io/downloads](http://continuum.io/downloads) in your web browser.
2. Download the Python 3 installer for Windows.
3. Install Python 3 using all the default settings, except check **Make Anaconda the default Python**.

#### Mac OS X

[Video Tutorial](https://www.youtube.com/watch?v=TcSAln46u9U)

1. Open [http://continuum.io/downloads](http://continuum.io/downloads) in your web browser.
2. Download the Python 3 installer for OS X.
3. Install Python 3 using all the default settings.

#### Linux

1. Open [http://continuum.io/downloads](http://continuum.io/downloads) in your web browser.
2. Download the Python 3 installer for Linux.  
   *(Installation requires using the shell. If you're not comfortable, request help at the workshop.)*
3. Open a terminal window.
4. Type `bash Anaconda3-` and press tab. The downloaded file name should appear. If not, navigate to the folder where you downloaded the file using, for example

```bash
   cd Downloads
```
Then try again.

5. Press enter and follow the prompts. Use the space key to scroll through text. Approve the license by typing `yes` and pressing enter. Accept the default file location by pressing enter. Type `yes` and press enter to prepend Anaconda to your `PATH` (this makes Anaconda the default Python).
6. Close the terminal window.


## Enumlib

To compile the code manually, clone the repository with the --recursive flag:
~~~
git clone --recursive https://github.com/msg-byu/enumlib.git
~~~
{: .bash}

We need to compile the symlib submodule before compiling enumlib.
Go to the enumlib/symlib/src directory:

~~~
cd enumlib/symlib/src
~~~
{: .bash}

Set an environment variable to identify your fortran compiler:
[in the bash shell, gfortran compiler]
export F90=gfortran
[the Makefile also recognizes ifort]

Then compile using the Makefile:
~~~
make
~~~
{: .bash}


(Alternatively, instead of setting the F90 environmental variable first, you may just specify the variable during the make: 

~~~
$ make F90=gfortran.)
~~~
{: .bash}
Next, make the enumeration library itself
~~~
$ cd ../../src
$ make
~~~
{: .bash}

Finally, to make a stand-alone executable for enumeration:
~~~
make enum.x
~~~
{: .bash}
It is possible to compile enumlib using conda on OSX and Linux. To do so use the command:

~~~
$ conda install --channel matsci enumlib
~~~
{: .bash}


Alternatively you would like to use the enumeration capabilities powered by Gus Hart’s enumlib or perform Bader charge analysis powered by the Bader analysis code of the Henkelmann group, please try installing these from source using the pmg command line tool as follows::
~~~
$ pmg config --install enumlib
$ pmg config --install blader 
~~~

## HyDGL
~~~
pip install git+https://github.com/hkneiding/HyDGL
~~~
{: .bash}

which installs HyDGL as a library to your Python installation or virtual environment.

Afterwards you can import the library with:
~~~
import HyDGL
~~~
{: .python





