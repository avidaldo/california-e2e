# Workflow: the California housing problem, step by step

Only what is done to solve this problem, in order. Every step links the [note](../notes/) that
explains the concept behind it; nothing is done twice or "the wrong way first".

1. [Framing the problem](01_framing.ipynb) — first look at the data, problem definition, RMSE as performance measure
2. [Exploratory data analysis](02_eda.ipynb) — geographic plots, correlations, capped values
3. [Train/test split](03_train_test_split.ipynb) — stratified split by income category
4. [Feature engineering decisions](04_feature_engineering.ipynb) — ratios, logarithms, geographic clusters
5. [Preprocessing pipeline](05_preprocessing_pipeline.ipynb) — the `ColumnTransformer` that implements those decisions
6. [Model selection](06_model_selection.ipynb) — three candidates compared with 10-fold cross-validation
7. [Hyperparameter tuning](07_hyperparameter_tuning.ipynb) — two rounds of randomized search; the test set is used once, here
8. [Neural network](08_neural_network.ipynb) — an MLP in PyTorch against the tuned random forest

`utils/` holds the code consolidated from steps 3 (`load_california.py`) and 5
(`housing_preprocessing.py`), plus the PyTorch training loop used in step 8 (`training_utils.py`).
Run the notebooks with this folder as the working directory so that `from utils... import ...`
and the `../data`, `../img` paths resolve.
