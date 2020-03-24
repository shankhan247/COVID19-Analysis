
import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import json
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
N_NEIGHBORS = 5
MIN_CASES = 1000
NORMALIZE = True
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_19-covid-Confirmed.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []
US = []
US_labels = []
for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    features.append(cases)
    targets.append(labels)
    #print(cases)
    if labels[0][1] == 'Italy':
        US.append(cases)
        US_labels.append(labels)

forecast = 1 # this will be the number of days we want to predict into the future
print(US)
US = np.concatenate(US,axis=0)
US_cases = np.sum(US,axis=0) # sum cases across all US locations
print(US_cases)
#print(US_cases)
US_features = np.reshape(US_cases,(-1,1))[:-forecast] # reshape array into proper size and remove values according to forecast
US_targets = US_cases[forecast:]
#print(US_targets.shape)
#print(US_features.shape)

f_train, f_test, t_train, t_test =  train_test_split(US_features, US_targets, test_size=0.1)

poly = PolynomialFeatures(degree=2) # define polynomial
p = poly.fit_transform(f_train) # transform training data as polynomial with degree 2
model2 = LinearRegression() # define linear model
model2.fit(p,t_train) # feed polynomial data into model (this is the training data we will use to generate the plot)
p_test = poly.fit_transform(f_test) # transform testing data
confidence2 = model2.score(p_test,t_test) # find confidence of model
print(confidence2)

# now we get the predicted case numbers for the next x days (ie forecast)
predict_features = np.reshape(US_cases,(-1,1))[-forecast:] # get the last numbers (ie forecast) of values from features 
predict_case2 = poly.fit_transform(predict_features)
predict_m2 = model2.predict(predict_case2)
print(predict_m2)

pred2 = model2.predict(p) # get the predicted values from training data (so we can line plot this and compare to actual case targets)
print(US_cases)

# plot the fitted curve
plt.scatter(f_train,t_train, s=15, marker='o', color='k')
plt.plot(np.sort(f_train, axis=0),np.sort(pred2), '--r')
plt.xlabel('Total Cases on a Day')
plt.ylabel('Predicted Total Cases After 3 Days')
plt.title('Relationship Between Changes of Total Corona Cases Within US')
plt.grid()
plt.show()

# plot cases
US_cases = np.append(US_cases, predict_m2)

# this loop predicts x amount of future cases 
num_pred = 10
for i in range(num_pred):
    pred_f = np.reshape(US_cases,(-1,1))[-forecast:]
    pred_Case = poly.fit_transform(pred_f)
    pred_m = model2.predict(pred_Case)
    US_cases = np.append(US_cases, pred_m)
print(US_cases)

num_days = len(US_cases) # number of days to plot
days = np.linspace(1,num_days,num_days,dtype=int)

# make the curve smooth (the next 3 lines code is from a forum post)
xnew = np.linspace(days.min(), days.max(), 300) 

spl = make_interp_spline(days, US_cases, k=3)  # type: BSpline
power_smooth = spl(xnew)

plt.plot(xnew, power_smooth, '--k')
s_known = plt.scatter(days[:-(forecast+num_pred)], US_cases[:-(forecast+num_pred)], s=15, marker='o', color='k')
s_predicted = plt.scatter(days[-(forecast+num_pred):], US_cases[-(forecast+num_pred):], s=20, marker='o', color='r')
plt.legend((s_known, s_predicted), ('known cases', 'predicted cases'))
plt.xlabel('Number of Days Since First Corona Case')
plt.ylabel('Total Corona Cases Predicted')
plt.title('Total Corona Cases Predicted Within the US Per Day')
plt.grid()
plt.show()

# compute rate of cases (per day)
n = len(US_cases)
rates = [0]
for i in range(n-1):
    case1 = US_cases[i]
    case2 = US_cases[i+1]
    diff = case2 - case1
    rates.append(diff)
plt.plot(days, rates, 'k')
#plt.scatter(days, rates, s=15, marker='o', color='r')
plt.grid()
plt.show()

