# California Housing Prices — an end-to-end machine learning project

This project is an adaptation of the [original by *Aurélien Géron*](https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb), from [the second chapter](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/ch02.html) of his book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/).

The goal is to predict the median house value of each California district from features such as location, number of rooms, population and median income. It is a classic dataset for regression and a paradigmatic example of a **supervised learning regression** problem, used here to walk through the whole process: from data loading and exploration to model training, tuning and evaluation.

## Two tracks

The material is split in two folders that serve different purposes:

| Folder | What it is | How to read it |
|---|---|---|
| [`workflow/`](workflow/) | **Only the steps applied to this problem**, in order, with the minimum explanation needed to follow them. Nothing is done twice or "the wrong way first". | Sequentially, `01` → `08`. Each step links the note that explains the concept behind it. |
| [`notes/`](notes/) | **One concept per notebook**: the explanations, alternatives, and small examples (the manual split, the imputation and encoding options, the leakage mistakes, grid vs. randomized search...). Most use the California data, but none depends on the workflow. | In any order, when a workflow step links to them or when a concept needs a refresher. |

The notes can also be read on their own, without the workflow, as a reference on data preparation, validation and the scikit-learn API.

## Workflow

1. [Framing the problem](workflow/01_framing.ipynb) — first look at the data, problem definition, performance measure
2. [Exploratory data analysis](workflow/02_eda.ipynb) — geographic plots, correlations, capped values
3. [Train/test split](workflow/03_train_test_split.ipynb) — stratified split, consolidated in `utils/load_california.py`
4. [Feature engineering decisions](workflow/04_feature_engineering.ipynb) — ratios, logarithms, geographic clusters
5. [Preprocessing pipeline](workflow/05_preprocessing_pipeline.ipynb) — the `ColumnTransformer`, consolidated in `utils/housing_preprocessing.py`
6. [Model selection](workflow/06_model_selection.ipynb) — linear regression vs. decision tree vs. random forest with cross-validation
7. [Hyperparameter tuning](workflow/07_hyperparameter_tuning.ipynb) — two rounds of randomized search; final evaluation on the test set
8. [Neural network](workflow/08_neural_network.ipynb) — regression with PyTorch, compared with the tuned random forest

## Notes

**Framing and evaluation**
- [Regression metrics: RMSE and MAE](notes/regression_metrics.ipynb)

**Understanding the data**
- [Looking for correlations](notes/correlation.ipynb)
- [Outliers and capped values](notes/outliers.ipynb)
- [Feature engineering: combining attributes](notes/feature_engineering.ipynb)

**Splitting and validation**
- [Training and test sets](notes/train_test_split.ipynb)
- [Stratified sampling](notes/stratified_sampling.ipynb)
- [Data leakage](notes/data_leakage.ipynb)
- [Cross-validation](notes/cross_validation.ipynb)

**Preprocessing**
- [Missing values](notes/missing_values.ipynb)
- [Encoding categorical features](notes/categorical_encoding.ipynb)
- [Feature scaling](notes/feature_scaling.ipynb)
- [Heavy-tailed distributions](notes/heavy_tailed_distributions.ipynb)

**The scikit-learn API**
- [Types of estimators](notes/sklearn_estimators.ipynb)
- [Pipelines](notes/pipelines.ipynb)
- [Custom transformers](notes/custom_transformers.ipynb)

**Feature construction**
- [Handling geographic coordinates](notes/geographic_features.ipynb)

**Model tuning**
- [Hyperparameter search](notes/hyperparameter_search.ipynb)

## Running the notebooks

The notebooks use relative paths (`../data/housing.csv`, `../img/`), so run each one with its own folder as the working directory, which is what Jupyter and VS Code do by default. The workflow notebooks import the shared code from `workflow/utils/`; the notes are self-contained.

Dependencies: pandas, NumPy, Matplotlib, seaborn, scikit-learn ≥ 1.4, SciPy, Pillow and, for the last workflow step, PyTorch.
