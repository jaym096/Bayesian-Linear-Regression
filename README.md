# Bayesian-Linear-Regression

This is an academic project done in the course **CSCI-B55 Machine Learning** at Indiana University.

Applied **Regularized Bayesian Linear Rgression** on various datasets and experimented with the regularization parameter to find the best
statistical model. Leveraged Bayesian model selection to find the parameters of the prior. Compared different approaches and concluded
that Bayesian model selection is a good method to obtain hyperparameters for statistical model.

The project consists of 2 code files (.py):\
_main.py_, _supporting_functions.py_

The pp2data folder containing all the data set. The code considers that the data folder is in the same directory as the code.

## HOW TO RUN THE CODE
    1. To run the code you only have to use the main.py file which takes two command line arguments
       -> filename: name of the data set file
       -> integer value (1,2): to decide which experiment to run
	            1 : experiment 1
	            2 : experiment 2
    2. Example: If I want to perform experiment 1 on yelp data set then:
                python main.py yelp_labelled.txt 1
