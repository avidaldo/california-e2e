# Notes: one concept per notebook

Independent notebooks that explain a single concept each, with the alternatives and the small
examples that the [workflow](../workflow/) deliberately leaves out. Most use the California
housing data (`../data/housing.csv`) because it is at hand, but none depends on the workflow code.
Read them in any order.

**Framing and evaluation**
- [Regression metrics: RMSE and MAE](regression_metrics.ipynb) — metrics vs. loss functions; why RMSE penalizes large errors

**Understanding the data**
- [Looking for correlations](correlation.ipynb) — Pearson coefficient, scatter matrix, what it misses
- [Outliers and capped values](outliers.ipynb) — capping, winsorizing, trimming; effect on mean and min-max scaling
- [Feature engineering: combining attributes](feature_engineering.ipynb) — ratio features and their correlations

**Splitting and validation**
- [Training and test sets](train_test_split.ipynb) — manual split, `train_test_split`, `random_state`
- [Stratified sampling](stratified_sampling.ipynb) — sampling bias, strata, `pd.cut`, `stratify=`
- [Data leakage](data_leakage.ipynb) — feature, target and model-selection leakage; validation set
- [Cross-validation](cross_validation.ipynb) — k-fold, shuffled folds, the scoring sign convention

**Preprocessing**
- [Missing values](missing_values.ipynb) — MCAR/MAR/MNAR, deletion, `SimpleImputer`, `KNNImputer`
- [Encoding categorical features](categorical_encoding.ipynb) — `OrdinalEncoder`, `LabelEncoder`, `OneHotEncoder`, `get_dummies`
- [Feature scaling](feature_scaling.ipynb) — min-max, standard and robust scaling; scaling the target
- [Heavy-tailed distributions](heavy_tailed_distributions.ipynb) — skewness, which models suffer, the log transform

**The scikit-learn API**
- [Types of estimators](sklearn_estimators.ipynb) — estimators, transformers, predictors; `fit`/`transform`/`predict`
- [Pipelines](pipelines.ipynb) — `Pipeline`, `make_pipeline`, `ColumnTransformer`
- [Custom transformers](custom_transformers.ipynb) — `FunctionTransformer`, `BaseEstimator`/`TransformerMixin`

**Feature construction**
- [Handling geographic coordinates](geographic_features.ipynb) — K-means regions, distance to centroid, RBF similarity

**Model tuning**
- [Hyperparameter search](hyperparameter_search.ipynb) — `GridSearchCV`, `RandomizedSearchCV`, iterating
