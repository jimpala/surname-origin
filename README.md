surname-origin
================

[![Build Status](https://travis-ci.com/jpyne17/surname-origin.svg?branch=master)](https://travis-ci.com/jpyne17/surname-origin)
[![Coverage Status](https://coveralls.io/repos/github/jpyne17/surname-origin/badge.svg?branch=master)](https://coveralls.io/github/jpyne17/surname-origin?branch=master)

Surname nationality origin predictor using a vanilla, character-level recurrent neural network implemented with PyTorch.

## Setup
- Create a virtualenv and activate.
- Run `make init` from the command line to install pip dependencies given in `requirements.txt`.
- Launch the `surname-origin-demo.ipynb` notebook for an interactive demo.
- Running `make app` from the command line will run the `main()` method in `./src/app.py`, which defaults to a 10,000 iteration training phase and the prediction given in the example below.
- Running `make test` from the command line will run the Pytest suite developed in the `./test` directory.

## Example
A typical prediction from a 10,000 iteration training phase yields:
```
Top 3 origin predictions for Konstantinidis:
(-0.35) Greek
(-3.19) Polish
(-3.31) Russian
```
With the top results ordered by best negative log-likelihood.