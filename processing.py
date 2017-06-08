# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:41:00 2017

@author: marios.koletsis
"""
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
from itertools import chain
from sklearn import decomposition
from sklearn import preprocessing

os.getcwd()

    
# setting the seed
seed = 42

# Loading the data and create the test and train sets 
nrows = None
date_convert = lambda x: pd.to_datetime(x, format='%Y-%m-%d')
df_start = pd.read_csv('train.csv', 
                       nrows=nrows,
                       parse_dates=['Date'],
                       date_parser=date_convert)

# Reading the store data
df_store = pd.read_csv('store.csv', nrows=nrows)

# spliting the data set in test and train

df_train = df_start.query('Date < "2015-06-14"')
df_test = df_start.query('Date > "2015-06-13"')

# make sure the cut is correct
len(df_test)+len(df_train) == len(df_start)

# Setting a variable to easily distinguish train (1) from submit (0) set
df_train['Set'] = 1
df_test['Set'] = 0


# Combine again train and test set
frames = [df_train, df_test]
df = pd.concat(frames)

# delete rows that sales are 0 regardless of the stores being open
df = df.loc[~((df['Open'] == 1) & (df['Sales'] == 0))]



# some feature creation (main data)
var_name = 'Date'
# create new features Take day of the month
df['DayOfMonth'] = pd.Index(df[var_name]).day
df['WeekOfYear'] = pd.Index(df[var_name]).week # week of year
df['Month'] = pd.Index(df[var_name]).month
df['Year'] = pd.Index(df[var_name]).year
df['DayOfYear'] = pd.Index(df[var_name]).dayofyear

# fill nas
df['DayOfMonth'] = df['DayOfMonth'].fillna(0)
df['WeekOfYear'] = df['WeekOfYear'].fillna(0)
df['Month'] = df['Month'].fillna(0)
df['Year'] = df['Year'].fillna(0)
df['DayOfYear'] = df['DayOfYear'].fillna(0)

# creating some features for the main DF file
df['salesPerCustomerPerDay'] = df['Sales']/df['Customers']

# StateHoliday column has values 0 & "0", So combine
df.StateHoliday.replace(0, "0", inplace=True)

df['StateHoliday'].describe()
# some feature engineering (store data)

## transform the competition related features to numerical
## from CompetitionOpenSinceYear and CompetitionOpenSinceMonth calculate the
## days in business for the competition

def convertCompetitionOpen(df):
    try:
        date = '{}-{}'.format(int(df['CompetitionOpenSinceYear']), int(df['CompetitionOpenSinceMonth']))
        return pd.to_datetime(date)
    except:
        return np.nan
        
df_store['CompetitionOpenSince'] = df_store.apply(lambda df: convertCompetitionOpen(df), axis=1)

### Convert competition open year and month to float
def convertPromo2(df):
    try:
        date = '{}{}1'.format(int(df['Promo2SinceYear']), int(df['Promo2SinceWeek']))
        return pd.to_datetime(date, format='%Y%W%w')
    except:
        return np.nan

df_store['Promo2Since'] = df_store.apply(lambda df: convertPromo2(df), axis=1)

#Deal with the 'PromoInterval' feature
## split the values in it
pi = df_store['PromoInterval'].str.split(',').apply(pd.Series, 1)
## form the split create new features
pi.columns = ['PromoInterval_A', 'PromoInterval_B', 'PromoInterval_C', 'PromoInterval_D']
## join the new features in the main df
df_store = df_store.join(pi)

## delete the initial feature
del df_store['PromoInterval']

# creating some features for the Store file
storeSalesPerDay = df.groupby([df['Store']])['Sales'].sum() / df.groupby([df['Store']])['Open'].count()
df_store = pd.merge(df_store, storeSalesPerDay.reset_index(name='storeSalesPerDay'), how='left', on=['Store'])

storeCustomersPerDay = df.groupby([df['Store']])['Customers'].sum() / df.groupby([df['Store']])['Open'].count()
df_store = pd.merge(df_store, storeCustomersPerDay.reset_index(name='storeCustomersPerDay'), how='left', on=['Store'])



##### For sales/customers
df['DateYM'] = df['Date'].apply(lambda x: (str(x)[:7]))

df['Sales'] = np.log(df['Sales'])
df['Customers'] = np.log(df['Customers'])

#  merging the datasets
df = pd.merge(df, df_store, how='left', on=['Store'])

# some new features
df['competitionDaysInbusiness'] = df['Date'] - df['CompetitionOpenSince']
df['promoActiveDays'] = df['Date'] - df['Promo2Since']

df['competitionDaysInbusiness'] = df['competitionDaysInbusiness'].fillna(0)
df['promoActiveDays'] = df['promoActiveDays'].fillna(0)


#df_original = df


## set categorical data (all) 
df[['Store','DayOfWeek','Open','Promo','SchoolHoliday','StateHoliday',
          'Set','DayOfMonth','WeekOfYear','Month','Year','StoreType',
          'Assortment','Promo2','PromoInterval_A','PromoInterval_B',
          'PromoInterval_C','PromoInterval_D']] \
= df[['Store','DayOfWeek','Open','Promo','SchoolHoliday','StateHoliday',
          'Set','DayOfMonth','WeekOfYear','Month','Year','StoreType',
          'Assortment','Promo2','PromoInterval_A','PromoInterval_B',
          'PromoInterval_C','PromoInterval_D']].apply(lambda x: x.astype('category'))

# Delete columns not going to use
toDelete = ['Date','DateYM','DayOfYear','WeekOfYear','DayOfMonth',
      'CompetitionOpenSinceYear','Promo2SinceWeek',
      'Promo2SinceYear','Promo2Since','CompetitionOpenSince',
      'CompetitionOpenSinceMonth']
df.drop(toDelete,axis=1, inplace=True,errors='ignore')


#one hot encoding (trainFile)
df_dum = pd.get_dummies(df[['DayOfWeek','Month','Year',
'StateHoliday','StoreType','Assortment']])

#  merging the datasets
df_dum = pd.DataFrame(df[['storeSalesPerDay','Set','Sales','Customers','Store','Open','Promo','Promo2',
'SchoolHoliday','storeCustomersPerDay','competitionDaysInbusiness','promoActiveDays'
]]).join(df_dum)


# timedelta to int
df_dum['competitionDaysInbusiness'] = (df_dum['competitionDaysInbusiness']/ np.timedelta64(1, 'D')).astype(int)
df_dum['promoActiveDays'] = (df_dum['promoActiveDays']/ np.timedelta64(1, 'D')).astype(int)


trainFile = df_dum.query('Set == "1"')
testFile = df_dum.query('Set == "0"')
testFile['Id'] = range(1,len(testFile)+1)


# write to CSV
trainFile.to_csv('train_dum.csv')
testFile.to_csv('test_dum.csv')


