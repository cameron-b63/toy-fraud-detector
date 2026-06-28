# Toy Fraud Detector
This repository encompasses my self-directed dive into the basics of ML and some common Python ML frameworks.

I got a lot of guidance from just asking a friend where to get started.

### Overview
This is not meant to be a robust fraud detector, hence the name **toy** fraud detector. This is a purely educational venture.

I found a credit card fraud dataset on Kaggle which appeared to be a good candidate for a simple binary classification model. I chose to implement the same simple pipeline using three different frameworks so I could learn each of them at a surface level, and then I performed a simple comparative analysis on the three models to get a feel for key metrics.

Before implementing that, I did some exploration with pandas and then just a simple logistic regression baseline with scikit.

### Project Structure
 - `/`
  - `README.md` *< you are here!*
  - `requirements.txt`
  - `common/`: functionality common across pipelines
  - `data/`: contains the csv for training and testing
  - `notebooks/`
    - `01_eda.py`: exploratory data analysis
    - `02_baseline.py`: logistic regression baseline
    - `03_pytorch_results`: annotated results for the pytorch version of the pipeline
  - `pytorch/`: pytorch version of the pipeline
  - `results/`: exported versions of models live here


### Setup
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate # or .venv\Scripts\activate on Windows
$ pip install -r requirements.txt
$ pip install -e .
```

### Usage
It's recommended to use each module as a python library from the project root.

Check `notebooks/` for more metrics than the simple evaluations.
#### PyTorch
Train: `python3 -m pytorch.train`
Evaluate: `python3 -m pytorch.evaluate`
