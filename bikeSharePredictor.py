# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 14:24:43 2017

@author: dgarg
"""
from __future__ import division
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot
from matplotlib.pyplot import plot as pp
from sklearn import model_selection
from sklearn import ensemble
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.model_selection import TimeSeriesSplit
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel, SelectPercentile

#load the data

#1 Read the data and test files.  Do the same preprocessing
mydata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')
 
mydata['date'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in mydata.datetime]
mydata['hour'] = [x.hour for x in mydata.date]
mydata['month'] = [x.month for x in mydata.date]
mydata['day'] = [x.day for x in mydata.date]
mydata['year'] = [x.year for x in mydata.date]
mydata['time'] = [x.month + 12*(x.year) for x in mydata.date]
mydata['dayofweek'] = [x.weekday() for x in mydata.date]
mydata['isMonday'] = [1 if(x == 0) else 0 for x in mydata['dayofweek']]
mydata['isTuesday'] = [1 if(x == 1) else 0 for x in mydata['dayofweek']]
mydata['isWednesday'] = [1 if(x == 2) else 0 for x in mydata['dayofweek']]
mydata['isThursday'] = [1 if(x == 3) else 0 for x in mydata['dayofweek']]
mydata['isFriday'] = [1 if(x == 4) else 0 for x in mydata['dayofweek']]
mydata['isSaturday'] = [1 if(x == 5) else 0 for x in mydata['dayofweek']]
mydata['isSunday'] = [1 if(x == 6) else 0 for x in mydata['dayofweek']]

# Add new features to fit to peaks at the start and end of workhours during workdays. 
mydata['hour_inverse_1'] = [(1/(abs(x-8) + 2))for x in mydata['hour']]
mydata['hour_inverse_2'] = [(1/(abs(x-17) + 15))for x in mydata['hour']]
#Add new features to fit peaks during late afternoons on weekends/holidays
mydata['hour_inverse_3'] = [(1/(abs(x-15) + 0.2))for x in mydata['hour']]
maxtime = max(mydata['time'])
mintime = min(mydata['time'])

testdata['date'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in testdata.datetime]
testdata['hour'] = [x.hour for x in testdata.date]
testdata['month'] = [x.month for x in testdata.date]
testdata['day'] = [x.day for x in testdata.date]
testdata['year'] = [x.year for x in testdata.date]
testdata['time'] = [x.month + 12*(x.year) for x in testdata.date]
testdata['dayofweek'] = [x.weekday() for x in testdata.date]
testdata['isMonday'] = [1 if(x == 0) else 0 for x in testdata['dayofweek']]
testdata['isTuesday'] = [1 if(x == 1) else 0 for x in testdata['dayofweek']]
testdata['isWednesday'] = [1 if(x == 2) else 0 for x in testdata['dayofweek']]
testdata['isThursday'] = [1 if(x == 3) else 0 for x in testdata['dayofweek']]
testdata['isFriday'] = [1 if(x == 4) else 0 for x in testdata['dayofweek']]
testdata['isSaturday'] = [1 if(x == 5) else 0 for x in testdata['dayofweek']]
testdata['isSunday'] = [1 if(x == 6) else 0 for x in testdata['dayofweek']]

testdata['hour_inverse_1'] = [(1/(abs(x-8) + 2))for x in testdata['hour']]
testdata['hour_inverse_2'] = [(1/(abs(x-17) + 15))for x in testdata['hour']]
testdata['hour_inverse_3'] = [(1/(abs(x-15) + 0.2))for x in testdata['hour']]
testdata['prediction_count'] = 0


features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity',
            'windspeed', 'dayofweek', 'hour']
            
for y in testdata['year'].unique():
    for m in testdata['month'].unique():
        test_slice = testdata.loc[(testdata['year'] == y) & (testdata['month'] == m)]
        #window start is the previous month
        windowstart = max(mintime, 12*y+m-1)
        #window stop is the next month
        windowend = min(maxtime, 12*y+m+1)
        
        mydata_slice = mydata.loc[(mydata['time'] >= windowstart) & (mydata['time'] <=windowend)]

        mydata_weekday = mydata_slice.loc[(mydata_slice['dayofweek']<5) & (mydata_slice['holiday'] == 0) ]
        mydata_weekend = mydata_slice.loc[(mydata_slice['dayofweek']>4) | (mydata_slice['holiday'] == 1) ]
        
        testdata_weekday = test_slice.loc[(test_slice['dayofweek']<5) & (test_slice['holiday'] == 0) ]
        testdata_weekend = test_slice.loc[(test_slice['dayofweek']>4) | (test_slice['holiday'] == 1) ]
        
                    
        ## Estimator for working days: 
        #mydata = mydata.loc[mydata['holiday'] ==0]
        features = ['temp', 'hour', 'hour_inverse_1', 'hour_inverse_2', 'humidity', 'windspeed', 'atemp', 'weather',
                    'isMonday', 'isTuesday', 'isWednesday', 'isThursday', 'isFriday']
                    
        poly = PolynomialFeatures(degree=6)
        train_ = poly.fit_transform(mydata_weekday[features])        
        test_ = poly.fit_transform(testdata_weekday[features])        
        # Use Ridge with normalization
        clf = linear_model.Ridge(normalize=True)
        clf.fit(train_, mydata_weekday['count'])
        sfm = SelectFromModel(clf, threshold=2)
        #sfm = SelectPercentile(clf, percentile=10)
        sfm.fit(train_, mydata_weekday['count'])
        train_n_features = sfm.transform(train_)
        test_n_features = sfm.transform(test_)
        #Fit again with trimmed features. 
        clf.fit(train_n_features, mydata_weekday['count'])
        #prediction
        prediction = clf.predict(test_n_features)
        prediction = [int(x) if(x>0) else 0 for x in prediction]
        testdata.loc[(testdata['year'] == y) & (testdata['month'] == m) & (testdata['dayofweek']<5) & (testdata['holiday'] == 0), 'prediction_count'] = prediction
        
        
        ## Estimator for weekends:
        features = ['temp', 'hour', 'humidity', 'windspeed', 'atemp', 'weather', 'hour_inverse_3', 'isSaturday', 'isSunday']
        
        poly = PolynomialFeatures(degree=6)
        train_ = poly.fit_transform(mydata_weekend[features])
        test_ = poly.fit_transform(testdata_weekend[features])
        
        clf = linear_model.Ridge(normalize=True)
        clf.fit(train_, mydata_weekend['count'])
        sfm = SelectFromModel(clf, threshold=0)
        #sfm = SelectPercentile(clf, percentile=10)
        sfm.fit(train_, mydata_weekend['count'])
        train_n_features = sfm.transform(train_)
        test_n_features = sfm.transform(test_)
        clf.fit(train_n_features, mydata_weekend['count'])
        prediction = clf.predict(test_n_features)
        prediction = [int(x) if(x>0) else 0 for x in prediction]
        pyplot.figure()
        pp(prediction)
        testdata.loc[(testdata['year'] == y) & (testdata['month'] == m) & ((testdata['dayofweek']>4) | (testdata['holiday'] == 1)), 'prediction_count'] = prediction
  

submission = pd.DataFrame({'datetime':testdata['datetime'],
                           'count': testdata['prediction_count']})
submission.to_csv('bikeSharing_windowing3m.csv', index=False)
    
