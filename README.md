# ML_Higgs

### Group: Diego Fiori, Paolo Colusso, Valerio Volpe
### Kaggle team name: LaVolpeilFioreEilColosso

The files are organised based on:
i) the process followed to implement the models,
ii) the Machine Learning algorithms being applied.

# (1) Preprocessing 

##"preprocessing.py"
Contains the functions used to clean the data. Specifically:<br />
-how to deal with missing values;
-creation of dummy variables;
-feature augmentation with interaction terms and polynomials;
-normalisation of data.



# (2) Generic functions

##"regression_tools.py"
Contains the generic functions used throughout the implementation of algorithms. Specifically:
-auxiliary functions for regression implementations;
-single steps of regression algorithms;
-extraction of a sample of the dataset;
-batch creation.

##"AIC.py"
Implements a subset selection method based on AIC. The method is implemented both for ridge and for logistic regression and constructs a series of models of increasing number of variables, greedily adding a new variable at each step. In the end the best of these models is selected using AIC. Contains the functions:
compare_aic_gradient_descent(y,tx,gamma,max_iter,threshold)
compare_aic_ridge(y,tx,lambda_)



# (3) Implementations

##"implementations.py"
Contains the implementations of the main machine learning algorithms we selected. The functions defined in this .py are:
-least_squares_GD
-least_squares_SGD
-least_squares
-ridge_regression
-ridge_regression_SGD
-lasso_regression_GD
-logistic_regression
-reg_logistic_regression
-logistic_regression_newton_method_demo



# (4) Cross Validations
Cross validation is used to set the values of hyperparameters and polynomial degrees in different regression models. The files which implement cross-validation are:
-"cross_validaion_logistic.py"
-"cross_validation_lasso.py"
-"cross_validation_ridge.py"



# (5) Estimate models and cross validate
-"test_lasso.py"
-"test_logistic_gd.py"
-"test_logistic_penalized.py"
-"test_AIC_logistic"




