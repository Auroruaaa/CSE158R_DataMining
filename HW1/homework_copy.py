import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import numpy
import random
import gzip
import math

import warnings
warnings.filterwarnings("ignore")

def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

answers = {} # Put your answers to each question in this dictionary

### Question 1

def feature(datum):
    # your implementation
    feat = datum['review_text'].count('!')
    return [1] + [feat]

X = [feature(d) for d in dataset]
Y = [d['rating'] for d in dataset]

model_1 = linear_model.LinearRegression(fit_intercept=False)
model_1.fit(X,Y)
theta = model_1.coef_
theta0, theta1 = theta

y_pred = model_1.predict(X)
mse = sum([x**2 for x in (Y-y_pred)]) / len(Y)

answers['Q1'] = [theta0, theta1, mse]

assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)


### Question 2

def feature(datum):
    review = datum['review_text']
    feat1 = len(review)
    feat2 = review.count('!')
    return [1] + [feat1] + [feat2]

X = [feature(d) for d in dataset]
Y = [d['rating'] for d in dataset]

model_2 = linear_model.LinearRegression(fit_intercept=False)
model_2.fit(X,Y)
theta = model_2.coef_
theta0, theta1, theta2 = theta
y_pred = model_2.predict(X)
mse = sum([x**2 for x in (Y-y_pred)]) / len(Y)

answers['Q2'] = [theta0, theta1, theta2, mse]

assertFloatList(answers['Q2'], 4)


### Question 3

def feature(datum, deg):
    # feature for a specific polynomial degree
    review = datum['review_text'].count('!')  
    f = [1]
    for i in range(deg):
        f.append(review**(i+1))
    return f

mses = []

for i in range(5):
    X = [feature(d,i+1) for d in dataset]
    Y = [d['rating'] for d in dataset]
    model_3 = linear_model.LinearRegression(fit_intercept=False)
    model_3.fit(X,Y)
    theta = model_3.coef_
    y_pred = model_3.predict(X)
    mse_current = sum([x**2 for x in (Y-y_pred)]) / len(Y)
    mses.append(mse_current)

answers['Q3'] = mses

assertFloatList(answers['Q3'], 5)# List of length 5


### Question 4

def feature(datum, deg):
    # feature for a specific polynomial degree
    review = datum['review_text'].count('!')  
    f = [1]
    for i in range(deg):
        f.append(review**(i+1))
    return f

mses = []

for i in range(5):
    X = [feature(d,i+1) for d in dataset]
    Y = [d['rating'] for d in dataset]
    length = len(X)//2

    model_4 = linear_model.LinearRegression(fit_intercept=False)
    model_4.fit(X[:length],Y[:length])
    
    residuals = model_4.predict(X[length:]) - Y[length:]
    mse = sum([x**2 for x in residuals]) / len(Y[:length])

    mses.append(mse_current)

answers['Q4'] = mses

assertFloatList(answers['Q4'], 5)


### Question 5

theta0 = sum(Y)/len(Y)
mae = sum(abs(x-theta0) for x in Y) / len(Y)

answers['Q5'] = mae

assertFloat(answers['Q5'])


### Question 6

f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))

X = [[1, d['review/text'].count('!')] for d in dataset]
y = [d['user/gender'] == 'Female' for d in dataset]

model_6 = linear_model.LogisticRegression()
model_6.fit(X,y)
predictions = model_6.predict(X)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BER = 1 - 1/2 * (TPR + TNR)

answers['Q6'] = [TP, TN, FP, FN, BER]

assertFloatList(answers['Q6'], 5)


### Question 7

X = [[1, d['review/text'].count('!')] for d in dataset]
y = [d['user/gender'] == 'Female' for d in dataset]

model_6 = linear_model.LogisticRegression(class_weight='balanced')
model_6.fit(X,y)
predictions = model_6.predict(X)

TP = sum([(p and l) for (p,l) in zip(predictions, y)])
FP = sum([(p and not l) for (p,l) in zip(predictions, y)])
TN = sum([(not p and not l) for (p,l) in zip(predictions, y)])
FN = sum([(not p and l) for (p,l) in zip(predictions, y)])
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
BER = 1 - 1/2 * (TPR + TNR)

answers["Q7"] = [TP, TN, FP, FN, BER]

assertFloatList(answers['Q7'], 5)

















