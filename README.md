# Example Data Science Project

## Description

This repository is a typical example of a data science project written in Python and is used for testing purposes when
we need to have a more truthful example of an algorithm in use by the Dutch Government.

The typical steps will be done:

0. exploratory data analysis
1. prepping data
2. training models
3. evaluating models
4. serving a result
5. monitoring

The steps do not follow a linear path, but actually loops back whenever one notices models do not have wanted
features.

## Tools used

Different frameworks are used for different stages in the project, for the exploratory data analyses `PyCaret` is used to
very quickly check a bunch of models and in which direction to build the "production" model later. For the "production"
model, `scikitlearn` is used to generate a not deep machine learning model. For the analysis on fairness and mitigation
of bias both `FairLearn` and `AIF360` are used. Possible extensitons are:

- For the analysis on data drift for the monitoring of the model `evidentlyai`

- For experiment tracking with `MLflow`.

This repository is inspired by the
[Thesis of Guusje Juijn](https://studenttheses.uu.nl/bitstream/handle/20.500.12932/43868/Thesis%20Guusje%20Final%20Version.pdf?sequence=1&isAllowed=y)
