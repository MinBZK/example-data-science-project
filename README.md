# Example Data Science Project

## Description

This repository contains a typical example of a data science project written in Python and is used for testing purposes when
we need to have a more realstic example of an algorithm in use by the Dutch Government.

The typical steps are:

0. exploratory data analysis
1. prepping data
2. training models
3. evaluating models
4. serving a result
5. monitoring

Building an algorithm is a non-linear process, because of dependencies and cycles in the above-mentioned steps. Because of this,
it is hard to define this in a standard project template. Therefore, I have chosen to make a single Python file and
comment out the chain-of-thought that one could have in order to e.g. mitigate bias or choosing a specific
model. So running `__main__.py` will make a model and does inference on it without anything to return.

## Kaggle

Before you can use this project you need a [kaggle](https://www.kaggle.com/) account and create a token. You
can store the token in ~/.kaggle/kaggle.json

## Tools used

Different frameworks are used for different stages in the project. For the exploratory data analyses `PyCaret` is used
to very quickly check a bunch of models to give an indication what kind of model to train for production purposes later
on. For the "production" model, `scikitlearn` is used to generate a non-deep machine learning model. For the analysis
on fairness and mitigation of bias both `FairLearn` and `AIF360` are used. Possible extensions are:

- For the analysis on data drift for the monitoring of the model `evidentlyai`, now just a very small part has been used

- For experiment tracking and logging with `MLflow`.

This repository is inspired by the
[Thesis of Guusje Juijn](https://studenttheses.uu.nl/bitstream/handle/20.500.12932/43868/Thesis%20Guusje%20Final%20Version.pdf?sequence=1&isAllowed=y)
