#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 01:59:34 2019

@author: jones
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'Importing the datasets'
df1 = pd.read_csv('student-mat.csv')
df2 = pd.read_csv('student-por.csv')
df3 = pd.concat([df1, df2])
df3.head()


'Data preprocessing and Exploratory analysis'
df3.drop_duplicates(["school","sex","age","address","famsize","Pstatus","Medu",
                     "Fedu","Mjob","Fjob","reason","nursery","internet"])
df3.columns
df3.describe()

df3.corr()

df3.info()

'Drop the columns which is not essentials for grade prediction'
df3 = df3.drop(['famsize', 'Pstatus', 'Fjob', 'Mjob'], axis=1)
df3 = df3.drop(['reason','traveltime', 'studytime', 'failures'], axis=1)
df3 = df3.drop(['schoolsup','famsup', 'paid', 'nursery', 'internet', 'freetime'], axis=1)
df3 = df3.drop(['higher', 'health'], axis=1)
df3.columns

'Visualizing the plots of the data according to the gender'
plt.pie(df3['sex'].value_counts().to_list(), 
        labels=['Female','Male'], colors=['crimson', 'slateblue'],
        autopct='%1.1f%%', startangle=90)
axis = plt.axis('equal')

'Visualizing the plots of the data according to the guardians nominal'
plt.pie(df3['guardian'].value_counts().to_list(),
        labels=['mother', 'father', 'other'],
        colors=['darkslateblue', 'mediumorchid', 'steelblue'],
        autopct='%.2f', startangle=90)
plt.axis('equal')


'Given the high correlation between different grades, drop G1 & G2'
df3 = df3.drop(['G1', 'G2'], axis=1)

'combine weekdays alcohol consumption with weekend alcohol consumption'
df3['Dalc'] = df3['Dalc'] + df3['Walc']

'combine mothers education with fathers education & call it parents education'
df3['Pedu'] = df3['Medu'] + df3['Fedu']

'combine goout and absences'
df3['goout'] = df3['goout'] + df3['absences']
df3 = df3.drop(['Walc','Medu','Fedu','absences'], axis=1)
df3.columns

'Getting dummies'
df3 = pd.get_dummies(df3, drop_first=True)
df3.info()


'define target variable and training and test sets'
X = df3.drop("G3",axis=1)
Y = df3["G3"]


'Splitting the dataset into the Training set and Test set'
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


'Fitting Multiple Linear Regression to the Training set'
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

' Predicting the Test set results'
y_pred = regressor.predict(X_test)

'Building Optimal Model using Backward Elimination'
import statsmodels.formula.api as sm
X_opt = X
regressor_OLS = sm.OLS(endog =Y, exog = X_opt).fit()
regressor_OLS.summary()

'''
Backward Eliminiation Process
Drop the variable which is not significant(p>0.05)
'''
X_opt = X.drop(['goout','activities_yes', 'address_U', 'school_MS', 'sex_M', 'guardian_mother'], axis=1)
regressor_OLS = sm.OLS(endog =Y, exog = X_opt).fit()
regressor_OLS.summary()


