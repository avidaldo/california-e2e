# California Housing Prices
## End-to-End Machine Learning Project

This notebook is an adaptation of the [original by *Aurélien Gerón*](https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb), from [the second chapter](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/ch02.html) of his book: [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition. Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)

This project demonstrates an end-to-end machine learning workflow using the California Housing Prices dataset. The goal is to predict housing prices based on various features such as location, number of rooms, and population.

It's a classic dataset for regression tasks and is widely used for educational purposes in machine learning. The project is structured to guide you through the entire process, from data loading and exploration to model training and evaluation; and it is a paradigmatic example of a **supervised learning regression** problem.



## Notebooks

1. [**Framing the Problem**](e2e010_framing.ipynb) — Problem definition, first look at the data, and performance metrics
2. [**Exploratory Data Analysis (EDA)**](e2e020_eda.ipynb) — Data visualization, correlations, and outlier analysis
3. [**Train/Test Split**](e2e025_train_test.ipynb) — Random and stratified sampling strategies
4. [**Feature Engineering**](e2e030_feature_engineering.ipynb) — Creating new features from existing ones

### Preprocessing
5. [**Missing Values**](e2e041_missing.ipynb) — Handling unavailable data with imputation strategies
6. [**Categorical Variables**](e2e042_categorical.ipynb) — Ordinal and one-hot encoding
7. [**Feature Scaling**](e2e043_scaling.ipynb) — Normalization, standardization, and heavy-tailed distributions

### Pipelines & Transformers
8. [**Pipelines**](e2e050_pipelines.ipynb) — Building preprocessing pipelines with scikit-learn
9. [**Custom Transformers**](e2e051_custom_transformers.ipynb) — Creating custom transformers for pipelines
10. [**Spatial Clustering**](e2e060_spatial_clustering.ipynb) — Handling geographic coordinates with K-means and RBF kernels

### Model Training & Evaluation
11. [**Model Evaluation**](e2e070_model_evaluation.ipynb) — Training models, cross-validation, and comparing performance
12. [**Hyperparameter Optimization**](e2e080_hyperparameters.ipynb) — Grid search and randomized search
13. [**Hyperparameter Tuning (Practice)**](e2e081_hyperparameters2.ipynb) — Hands-on hyperparameter tuning exercise
14. [**Neural Networks**](e2e090_neural_network/e2e090_neural_network.ipynb) — Regression with PyTorch
