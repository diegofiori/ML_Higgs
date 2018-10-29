# ML_Higgs

### Group: Diego Fiori, Paolo Colusso, Valerio Volpe
### Kaggle team name: LaVolpeilFioreEilColosso

The files are organised based on:
i) the process followed to implement the models,
ii) the Machine Learning algorithms being applied.

#Preprocessing 
"preprocessing.py"
Contains the functions used to clean the data. Specifically:
-how to deal with missing values;
-creation of dummy variables;
-feature augmentation with interaction terms and polynomials;
-normalisation of data.

#Generic functions
"regression_tools.py"
Contains the generic functions used throughout the implementation of algorithms. Specifically:
-auxiliary functions for regression implementations;
-single steps of regression algorithms;
-extraction of a sample of the dataset;
-batch creation.

#Implementations
"implementations.py"
Contains the implementations of the main machine learning algorithms we selected. The functions defined in this .py are:
-least_squares_GD
-least_squares_SGD
-least_squares
-ridge_regression
-ridge_regression_SGD
-lasso_regression_GD
-logistic_regression
-reg_logistic_regression

#Cross Validations
Cross validation is used to set the values of hyperparameters and polynomial degrees in different regression models. The 
-"cross_validaion_logistic.py"
-"cross_validation_lasso.py"
-"cross_validation_ridge.py"

#Estimate models and cross validate
"test_lasso.py"
"test_logistic_gd.py"
"test_logistic_penalized.py"



