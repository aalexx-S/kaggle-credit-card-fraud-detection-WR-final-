#!/bin/bash

git clone https://github.com/scikit-learn-contrib/imbalanced-learn.git
cd imbalanced-learn
pip install .
cd ..

git clone https://github.com/akazs/pySMOTE.git
cd pySMOTE
pip install .
