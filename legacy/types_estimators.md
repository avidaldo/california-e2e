# Types of Estimators in scikit-learn

It's important to clarify certain terms used in scikit-learn:

- **Estimators**: any object that can estimate parameters based on a dataset is called an estimator. The estimation itself is performed through the **`fit()`** method.

    - **Transformers**: estimators that can also transform data using the **`transform()`** method. For example, `SimpleImputer` is a transformer: it estimates values with `fit()` and imputes them with `transform()`.
        - Scalers: transformers that scale data.
        - Imputers: transformers that impute missing values.
        - Encoders: transformers that encode categorical variables.
        - Dimensionality reducers: transformers that reduce the number of variables.
        - ...

    - **Predictors**: those estimators that are capable of making predictions based on a dataset. For example, the linear regression model is a predictor: it estimates hyperparameters with `fit()` and makes predictions with **`predict()`**.
        - Classifiers: predictors that predict categorical labels.
        - Regressors: predictors that predict continuous values.
        - Clusterers: predictors that group data into clusters.
        - ...

The term "predictor" can be confusing as it is also used, in general, to refer to the *features* or independent variables of a model, and sometimes only for those variables that are indeed **predictive**, leaving out those characteristics that do not have predictive capacity.

Also, the term "transformer" should not be confused with the popular "transformer" neural network architecture, which is the basis on which language models like GPT are built.

https://scikit-learn.org/stable/developers/develop.html

<!-- TODO: Explain separately the scikit-learn design principles in detail with examples (it's a good way to work on software engineering concepts in Python)-->
