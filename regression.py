import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

######################################################################################################################
# What is Regression, What Does It Do?
######################################################################################################################

######################################################################################################################
# Regression
######################################################################################################################

'''
In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between 
a dependent variable (often called the 'outcome' or 'response' variable, or a 'label' in machine learning parlance) and 
one or more independent variables (often called 'predictors', 'covariates', 'explanatory variables' or 'features'). The 
most common form of regression analysis is linear regression, in which one finds the line (or a more complex linear 
combination) that most closely fits the data according to a specific mathematical criterion. If you don't know, 
today I will tell you in detail about regression, one of the most important types of data analysis, and how regression 
analysis works.
'''

######################################################################################################################
#What Does Regression Do?
######################################################################################################################

'''
We can say that regression analysis is used to estimate the value of the dependent variable depending on the independent
 variables.
'''

######################################################################################################################
#Linear Regression
######################################################################################################################

'''
Linear regression is used to model the relationship between two variables and estimate the value of a response by using 
a line-of-best-fit.

Regression analysis formula:

Formula Y = MX + b

Y is the dependent variable of the regression equation. M is the slope of the regression equation. X is the dependent 
variable of the regression equation. b is the constant of the equation.
If we want to examine it on an example. Advertising data sales (in thousands of units) for a particular product 
advertising budgets (in thousands of dollars) for TV, radio, and newspaper media
'''


# https://www.kaggle.com/datasets/bumba5341/advertisingcsv

df = pd.read_csv("Advertising.csv")

df.head()

df.shape

# Let's try to estimate the advertising fees spent on TV ads based on product sales.

X = df[["TV"]]
y = df[["Sales"]]

# Model

reg_model = LinearRegression().fit(X, y)

# constant (b - bias)
b = reg_model.intercept_[0]

# coefficient of TV (M)
M = reg_model.coef_[0][0]

print("Linear regression parameters at : b = {0}, M = {1}".format(b, M))


######################################################################################################################
# Prediction
######################################################################################################################

'''
The most common use of regression analysis is to predict future opportunities and threats. For this example, 
let's estimate what the sales would be if there were 150 units of spending on TV ads.
'''


reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

# Functional:
new_data = [150]

new_data = pd.DataFrame(new_data,columns=['TV'])

reg_model.predict(new_data)


######################################################################################################################
# Visualization of the Model
######################################################################################################################

# Visualization of the Model
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")
g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Sales")
g.set_xlabel("TV")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


######################################################################################################################
# Evaluating Forecast Success
######################################################################################################################


######################################################################################################################
# MSE
######################################################################################################################

'''
In statistics, the mean squared error (MSE) or mean squared deviation (MSD) of an estimator (of a procedure for 
estimating an unobserved quantity) measures the average of the squares of the errors that is, the average 
squared difference between the estimated values and the actual value. MSE is a risk function, corresponding to 
the expected value of the squared error loss. The MSE is a measure of the quality of an estimator. As it is derived 
from the square of Euclidean distance, it is always a positive value that decreases as the error approaches zero.
'''

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)


######################################################################################################################
# RMSE
######################################################################################################################

'''
The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) is a frequently used measure of the differences
 between values (sample or population values) predicted by a model or an estimator and the values observed. The RMSD
  represents the square root of the second sample moment of the differences between predicted values and observed values
   or the quadratic mean of these differences. RMSD is always non-negative, and a value of 0 (almost never achieved in 
   practice) would indicate a perfect fit to the data.
   '''

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

######################################################################################################################
# MAE
######################################################################################################################
'''
mean absolute error (MAE) is the mean of the absolute differences between the actual values and the estimated values
'''

# MAE
mean_absolute_error(y, y_pred)


######################################################################################################################
# R-squared
######################################################################################################################

'''
R-squared is a goodness-of-fit measure for linear regression models. This statistic indicates the percentage of the 
variance in the dependent variable that the independent variables explain collectively. R-squared measures the strength
of the relationship between your model and the dependent variable on a convenient 0–100% scale.
'''

# R-squared
reg_model.score(X, y)


######################################################################################################################
# Multiple Linear Regression
######################################################################################################################

'''
Multiple linear regression is used to estimate the relationship between two or more independent variables and one 
 dependent variable.
Let's create a multiple regression model with all the variables in the Advertising dataset.
'''


X = df[['TV','Radio','Newspaper']]
y = df[["Sales"]]


# Model
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)


# constant (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_

######################################################################################################################
# Prediction
######################################################################################################################

'''
What is the expected value of the sale based on the following observation values?
TV: 30
radio: 10
newspaper: 40
'''

# Prediction
new_data = [[30], [10], [40]]

new_data = pd.DataFrame(new_data).T
new_data.columns =['TV', 'Radio', 'Newspaper']

reg_model.predict(new_data)


######################################################################################################################
# Evaluating Forecast Success
######################################################################################################################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN R-squared
reg_model.score(X_train, y_train)

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test RKARE
reg_model.score(X_test, y_test)

######################################################################################################################
# CV (Cross-validation)
######################################################################################################################

'''
Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: 
a model that would just repeat the labels of the samples that it has just seen would have a perfect score but 
would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is
common practice when performing a (supervised) machine learning experiment to hold out part of the available data 
as a test set X_test, y_test. Note that the word "experiment" is not intended to denote academic use only, 
because even in commercial settings machine learning usually starts out experimentally. Here is a flowchart of 
typical cross validation workflow in model training. The best parameters can be determined by grid search techniques.
'''

# 10-Fold CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                  X,
                                  y,
                                  cv=10,
                                  scoring="neg_mean_squared_error")))

######################################################################################################################
# What to Consider in Regression Analysis?
######################################################################################################################


######################################################################################################################
# Outliers
######################################################################################################################

'''
Outliers are defined as abnormal values that do not fit the normal distribution in a dataset and have the potential to 
significantly distort any regression model. Therefore, outliers must be handled carefully to gain accurate insights 
from the data. Usually, data collected from the real world consists of various observations of various characteristics, 
and many of the values may be misplaced. Whatever the reason for the outliers, the presence of outliers negatively 
affects the analysis result.
'''


#Function that sets upper and lower limits for outliers.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Function that recompiles outliers
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    # Are there any outliers? func prepared to answer the question. It outputs true or folse.
    def check_outlier(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False


# To access indexes of outliers, index= true must be set
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


# Let's find outliers

grab_outliers(df, 'Newspaper', index=False)

#Let's suppress outliers.

replace_with_thresholds(df, 'Newspaper')


######################################################################################################################
#Multicollinearity
######################################################################################################################

'''
Multicollinearity occurs when two or more independent variables are highly correlated in a regression model. This means 
that in a regression model one independent variable can be predicted from another independent variable.
'''

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5,cmap='RdBu', fmt= '.1f',ax=ax, vmin=-1)
plt.show()

#There is no multicollinearity for this example.


######################################################################################################################
# Heteroskedasticity
######################################################################################################################

'''
In statistics, heteroskedasticity (or heteroscedasticity) happens when the standard errors of a variable, monitored 
over a specific amount of time, are non-constant. When observing the graph of residuals, a fan or cone shape indicates 
the presence of varying variance. In statistics, varying variance is seen as a problem because regressions involving 
ordinary least squares (OLS) assume that residuals are drawn from a population with constant variance. 
Heteroskedasticity is a violation of the assumptions for linear regression modeling, and so it can impact the validity 
of regression
'''

g = sns.regplot(x=X['TV'], y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")
g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Sales")
g.set_xlabel("TV")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

'Heteroskedasticity for this example is seen in the figure.'
