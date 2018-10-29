# ML_Higgs

### Machine Learning CS-433: Project 1
### Group: Diego Fiori, Paolo Colusso, Valerio Volpe
### Kaggle team name: LaVolpeilFioreEilColosso

The files are organised based on:<br />
i) the process followed to implement the models,<br />
ii) the Machine Learning algorithms being applied.<br />

# (1) Preprocessing 

### "preprocessing.py" <br />
Contains the functions used to clean the data. Specifically:<br />
-how to deal with missing values;<br />
-creation of dummy variables;<br />
-feature augmentation with interaction terms and polynomials;<br />
-normalisation of data.<br />



# (2) Generic functions

### "regression_tools.py" <br />
Contains the generic functions used throughout the implementation of algorithms. Specifically:<br />
-auxiliary functions for regression implementations: <br />
-single steps of regression algorithms;<br />
-extraction of a sample of the dataset;<br />
-batch creation.<br />

### "AIC.py"<br />
Implements a subset selection method based on AIC. The method is implemented both for ridge and for logistic regression and constructs a series of models of increasing number of variables, greedily adding a new variable at each step. In the end the best of these models is selected using AIC. Contains the functions:
compare_aic_gradient_descent(y,tx,gamma,max_iter,threshold)<br />
compare_aic_ridge(y,tx,lambda_)<br />



# (3) Implementations

### "implementations.py"<br />
Contains the implementations of the main machine learning algorithms we selected. The functions defined in this .py are:<br />
-least_squares_GD<br />
-least_squares_SGD<br />
-least_squares<br />
-ridge_regression<br />
-ridge_regression_SGD<br />
-lasso_regression_GD<br />
-logistic_regression<br />
-reg_logistic_regression<br />
-logistic_regression_newton_method_demo<br />



# (4) Cross Validations
Cross validation is used to set the values of hyperparameters and polynomial degrees in different regression models. The files which implement cross-validation are: <br />
-"cross_validaion_logistic.py"<br />
-"cross_validation_lasso.py"<br />
-"cross_validation_ridge.py"<br />



# (5) Estimate models 
The following files run algorithms implementing different machine learning algorithms from the data loading phase to the final csv creation.<br />
-"test_lasso.py"<br />
-"test_logistic_penalized.py"<br />
-"test_logistic_penalized-cross.py"<br />
-"test_logistic_newton.py"<br />
-"test_logistic_gd.py"<br />
-"test_AIC_logistic.py"<br />

A few functions, such as batch_iter, were taken from the helpers of the lab session of the course.

