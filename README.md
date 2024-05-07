# Python Project Template

## Description

This repository is a typical example of a data science project written in Python and is used for testing purposes when
we need to have a more truthful example of an algorithm in use by the Dutch Government.

The typical steps will be done:

1. prepping data
2. training models
3. evaluating models
4. serving a result
5. monitoring

When creating a new Repository select this template repository as the base.

## Kaggle

Before you can use this project you need a [kaggle](https://www.kaggle.com/) account and create a token. You
can store the token in ~/.kaggle/kaggle.json

* change the owners in the the .github/CODEOWNERS
* run a global rename command where you rename new_name to your project name
  * macos: `find . -type f -not -path "./.git/*" -exec  sed -i '' "s/python_project/new_name/g" {} \;`
  * linux: `find . -type f -not -path "./.git/*" -exec  sed -i "s/python_project/new_name/g" {} \;`
* rename the python_project/ folder to your project name
* change author and name in pyproject.toml
* change labels in Dockerfile to appropriate values
* Verify the License used
* Change publiccode.yml to your needs
