# Machine Learning Boilerplate

The goal of this repository is to be a starting plate for building out a well organized python project for machine learning. I've seen so many projects that are just a jumble of code and libraries and it's hard to follow and extend. This is an attempt to make it easier to get started and build out a project.

# Table of Contents

If you would like to skip to a specific section, here is a table of contents.

* [Environment Setup](#environment-setup)
* [Project Organization](#project-organization)

# Environment Setup

Your local development environment is an important first place to start. Often if you do not have your libraries isolated with the correct python version and dependencies, you will run into issues. This section will walk you through setting up your environment.

## GPU/CUDA

Out of scope for now, just making sure I remember to add this.

## PyEnv

Install pyenv with homebrew os OSX or your package manager of choice on Linux.

OSX:

```bash
$ brew install pyenv
```

Ubuntu:

```bash
$ curl https://pyenv.run | bash
```

Add the following to your ~/.bashrc file

```bash
# pyenv
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

Restart your shell to make sure the changes take place. Then install the version of python you would like to use. We're using 3.6.1 for this project.

```bash
$ pyenv install 3.10.8 # install the python version
$ pyenv global 3.10.8 # make this your global python version
```

Make sure your python version is correct.

```bash
$ which python # See where python is installed
$ python --version # Verify the version
```

## Virtualenv

After we verify we have the correct version of python installed, we can create a virtual environment to install our dependencies. This will keep our dependencies isolated from the rest of our system. I personally like to use virtualenv, but conda is also a good option.

Install virtualenv with pip.

```bash
$ pip install virtualenv # install virtualenv
```

Create the virtual environment. This will create a directory in your home directory called .my_project_name_venv. You can name this whatever you want. Python as well as all the packages you install with pip will go in this directory. It is easy to switch between environments and python versions, so you can have one for each project you are working on.

```bash
$ python -m venv ~/.my_project_name_venv
```

Activate the virtual environment. You should now see the name of the virtual environment in your terminal prompt.

```bash
$ source ~/.my_project_name_venv/bin/activate
```

Verify that python is now the one from your virtual environment.

```bash
$ which python
```

## Add path to your editor of choice

I use visual studio code, so I will add the path to my virtual environment to my settings.json file. This will allow me to use the python interpreter from my virtual environment. You can also do this in the settings UI by searching for "python" and going to "Default Interpreter Path".

```json
{
    "python.defaultInterpreterPath": "/Users/<username>/.venv3.10.6/bin/python",
}
```

# Project Organization

There are a few key scripts to kick things off:

