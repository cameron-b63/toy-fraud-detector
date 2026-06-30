# Toy Fraud Detector
This repository encompasses my self-directed dive into the basics of ML and some common Python ML frameworks.

I got a lot of guidance from just asking a friend where to get started.

### Overview
This is not meant to be a robust fraud detector, hence the name **toy** fraud detector. This is a purely educational venture.

I found a credit card fraud dataset on Kaggle which appeared to be a good candidate for a simple binary classification model. I chose to implement the same simple pipeline using three different frameworks so I could learn each of them at a surface level, and then I performed a simple comparative analysis on the three models to get a feel for key metrics.

Before implementing that, I did some exploration with pandas and then just a simple logistic regression baseline with scikit.

### Findings
The dataset used here is a tiny, severely imbalanced set with rather few features. PyTorch had the best out-of-the-box cooperability, with a recall of 1.0 for fraudulent charges only tuning for the class imbalance. Tensorflow, on the other hand, produced a model with a recall of 0.0, indicating a more involved tuning process. JAX met somewhere in the middle, with a recall of 0.75 but a precision of around 0.49. Tuning may help here, but it seems that the model just isn't very confident, with a serious cluster of probabilities near the threshold value of 0.5.

In terms of API and personal preference, I felt the best about JAX and PyTorch, and didn't really love the way tensorflow's keras felt to work with. JAX in particular has parity with my mental model of machine learning, and I felt that it provided the best balance between expressibility and verbosity. PyTorch was powerful, and felt quite approachable, but I worry (baselessly, I will admit; easy syntax simply inspires this fear) that its use cases are somewhat limited as the complexity of a pipeline grows.

I feel as though I'm walking out of this weekend project with a fairly strong understanding of the fundamentals of machine learning in Python. I feel as if I have enough foundational vocabulary and understanding to be teachable. I still have plenty more to learn about the semantics of multiclass classification, regression models, and unsupervised learning, but I think I have somewhere to start.

### Project Structure
 - `/`
  - `README.md` *< you are here!*
  - `requirements.txt`
  - `common/`: functionality common across pipelines
  - `data/`: contains the csv for training and testing
  - `jax_m/`: JAX implementation
  - `notebooks/`
    - `01_eda.py`: exploratory data analysis
    - `02_baseline.py`: logistic regression baseline
    - `03_pytorch_results`: annotated results for the pytorch version of the pipeline
    - `04_tensorflow_results`: annotated results for the keras implementation
    - `05_jax_results`: annotated results for the JAX/flax/optax implementation
  - `pytorch/`: pytorch version of the pipeline
  - `results/`: exported versions of models live here
  - `tf`: Keras implementation with tensorflow


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
