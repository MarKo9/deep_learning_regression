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
from scipy.stats.stats import pearsonr
from itertools import chain

#++++++++++++++ Loading the data and create the test and train sets ++++++++++

nrows = None
date_convert = lambda x: pd.to_datetime(x, format='%Y-%m-%d')
df_start = pd.read_csv('train.csv', 
                       nrows=nrows,
                       parse_dates=['Date'],
                       date_parser=date_convert)
                       
                       

# Reading the store data
df_store = pd.read_csv('store.csv', nrows=nrows)

# below is for loading the test set as well

nrows = nrows
df_test = pd.read_csv('test.csv', 
                        nrows=nrows,
                        parse_dates=['Date'],
                        date_parser=date_convert)    
# see the data 
df_start.head()
df_store.head()
df_test.head()
# spliting the data set in test and train based on the 6 weeks prediction interval

df_train = df_start.query('Date < "2015-06-14"')
df_submit = df_start.query('Date > "2015-06-13"')

# make sure the cut is correct
len(df_submit)+len(df_train) == len(df_start)

### Setting a variable to easily distinguish train (1) from submit (0) set
df_train['Set'] = 1
df_submit['Set'] = 0


### Combine train and test set
frames = [df_train, df_submit]
df = pd.concat(frames)


features_x = ['Store', 'Date', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday']
features_y = ['Sales']


# ++++++++++++++++ data cleaning (test data)  +++++++++++++++++

### Remove rows where store is open, but no sales.

df = df.loc[~((df['Open'] == 1) & (df['Sales'] == 0))]



# ++++++++++++++++ feature creation (test data) +++++++++++++++++

var_name = 'Date'
# create new features Take day of the month
df[var_name + 'DayOfMonth'] = pd.Index(df[var_name]).day
df[var_name + 'WeekOfYear'] = pd.Index(df[var_name]).week # week of year
df[var_name + 'Month'] = pd.Index(df[var_name]).month
df[var_name + 'Year'] = pd.Index(df[var_name]).year
df[var_name + 'DayOfYear'] = pd.Index(df[var_name]).dayofyear

# fill nas
df[var_name + 'DayOfMonth'] = df[var_name + 'DayOfMonth'].fillna(0)
df[var_name + 'WeekOfYear'] = df[var_name + 'WeekOfYear'].fillna(0)
df[var_name + 'Month'] = df[var_name + 'Month'].fillna(0)
df[var_name + 'Year'] = df[var_name + 'Year'].fillna(0)
df[var_name + 'DayOfYear'] = df[var_name + 'DayOfYear'].fillna(0)


# add the new features on the "features_x" list and delete the 'Date' 
features_x.remove(var_name)
features_x.append(var_name + 'DayOfMonth')
features_x.append(var_name + 'WeekOfYear')
features_x.append(var_name + 'Month')
features_x.append(var_name + 'Year')
features_x.append(var_name + 'DayOfYear')


# StateHoliday column has values 0 & "0", So combine
df.StateHoliday.replace(0, "0", inplace=True)

# holidays to category type
df['StateHoliday'] = df['StateHoliday'].astype('category')#.cat.codes

## set categorical data
df[['Set','StateHoliday','SchoolHoliday','DayOfWeek','Store','Open','Promo']] \
= df[['Set','StateHoliday','SchoolHoliday','DayOfWeek','Store','Open','Promo']].apply(lambda x: x.astype('category'))

## transform to categorical variables
df[['DateDayOfMonth','DateWeekOfYear','DateMonth','DateYear','DateDayOfYear','Open']] \
= df[['DateDayOfMonth','DateWeekOfYear','DateMonth','DateYear',
      'DateDayOfYear','Open']].apply(lambda x: x.astype('category'))

# ++++++++++++++++ some feature engineering (store data)  +++++++++++++++++

## transform the competition related features to numerical

#### from CompetitionOpenSinceYear and CompetitionOpenSinceMonth calculate the
#### days in business for the competition

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



#========== Deal with the 'PromoInterval' feature
## split the values in it
s = df_store['PromoInterval'].str.split(',').apply(pd.Series, 1)
# form the split create new features
s.columns = ['PromoInterval_A', 'PromoInterval_B', 'PromoInterval_C', 'PromoInterval_D']
# join the new features in the main df
df_store = df_store.join(s)

# delete the initial feature
del df_store['PromoInterval']

#========== Deal with the 'PromoInterval'/ feature 


# ============== New features =============

# creating some features for the stores file

# creating some features for the main DF file
df['salesPerCustomerPerDay'] = df['Sales']/df['Customers']


# creating some features for the Store file
storeSalesPerDay = df.groupby([df['Store']])['Sales'].sum() / df.groupby([df['Store']])['Open'].count()
df_store = pd.merge(df_store, storeSalesPerDay.reset_index(name='storeSalesPerDay'), how='left', on=['Store'])

storeCustomersPerDay = df.groupby([df['Store']])['Customers'].sum() / df.groupby([df['Store']])['Open'].count()
df_store = pd.merge(df_store, storeCustomersPerDay.reset_index(name='storeCustomersPerDay'), how='left', on=['Store'])



# ============== New features to be used later =============

##### For sales/customers


# Sales

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

ax1 = df["Sales"].plot(kind='hist',bins=70,xlim=(0,15000),ax=axis1)
ax1.set(xlabel = "Sales")
ax2 = df["Customers"].plot(kind='hist',bins=70,xlim=(0,3000),ax=axis2)
ax2.set(xlabel = "Customers")
sns.plt.suptitle('Distribution for sales and customers')

# see for not store that are closed
len(df[["Sales","Customers"]].query('Sales == 0'))
len(df[["Sales","Customers"]].query('Customers == 0'))


df_StoreOpen = df.query('Open == 1')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

ax1 = df_StoreOpen["Sales"].plot(kind='hist',bins=70,xlim=(0,15000),ax=axis1)
ax1.set(xlabel = "Sales")
ax2 = df_StoreOpen["Customers"].plot(kind='hist',bins=70,xlim=(0,3000),ax=axis2)
ax2.set(xlabel = "Customers")
sns.plt.suptitle('Distribution for sales and customers (Only for Open Stores)')

# calculate correlation
df_StoreOpen_SC = df_StoreOpen[["Sales","Customers"]]
pearsonr(df_StoreOpen["Sales"],df_StoreOpen["Customers"])

df_StoreOpen_SC.describe()

pd.plotting.scatter_matrix(df_StoreOpen_SC, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
plt.suptitle('Sales Vs Customers')


# after scaling
df['Sales_log'] = np.log(df['Sales'])
df['Customers_log'] = np.log(df['Customers'])

df_StoreOpen = df.query('Open == 1')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

ax1 = df_StoreOpen["Sales_log"].plot(kind='hist',bins=70,ax=axis1)
ax1.set(xlabel = "Sales")
ax2 = df_StoreOpen["Customers_log"].plot(kind='hist',bins=70,ax=axis2)
ax2.set(xlabel = "Customers")
sns.plt.suptitle('Distribution for sales and customers (Only Open Stores, log)')


#Find out what is going on with the closed stores

fig, (axis1) = plt.subplots(1,1,figsize=(15,5))
ax = sns.countplot(x='Open',hue='DayOfWeek', data=df,palette="husl", ax=axis1)
ax.set(xlabel='Day of week', ylabel='Open')
sns.plt.title('Days the stores were closed per day of week')

######## getting the some insights arount Sundays
df_sun_open = pd.DataFrame(df[['Store','Open','DayOfWeek']].query('DayOfWeek == 7 & Open == 1'))
# the list with the stores open on Sundays
TopSunOpen =pd.DataFrame(df_sun_open.groupby('Store').count().query('Open != 0')).reset_index()['Store']#.values.tolist()
TopSunOpen
# stores and the number of sundays beeing open
pd.DataFrame(df_sun_open.groupby('Store').count().query('Open != 0')).reset_index()[['Store','Open']]

######## getting  some insights arount the other days
df_wekdays_closed = pd.DataFrame(df[['Store','Open','DayOfWeek','StateHoliday']].query('DayOfWeek != 7 & Open == 0'))
# the list with the stores closed on weekdays (seems to be every one so does not make sense to take into count)
len(pd.DataFrame(df_wekdays_closed.groupby('Store').count()).reset_index()['Store'])
# the list with the stores slist on weekdays but not on national holidays (0) 
len(pd.DataFrame(df_wekdays_closed.groupby('Store').count().query('StateHoliday == 1')).reset_index()['Store'])

# Promo related analysis
# Plot average sales & customers with/without promo
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

ax1 =sns.barplot(x='Promo', y='Sales', data=df_StoreOpen, ax=axis1)
ax1.set(xlabel='Promo', ylabel='Sales')
ax2 =sns.barplot(x='Promo', y='Customers', data=df_StoreOpen, ax=axis2)
ax2.set(xlabel='Promo', ylabel='Customers')
sns.plt.suptitle('Promo and the effect on Customers and Sales')

# calculate sales per customer per day for promo days and no promo days
df_StoreOpen.query('Promo == 1')['Sales'].sum()/df_StoreOpen.query('Promo == 1')['Customers'].sum()
df_StoreOpen.query('Promo == 0')['Sales'].sum()/df_StoreOpen.query('Promo == 0')['Customers'].sum()


# Holidays related analysis
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

ax1 = sns.barplot(x='StateHoliday', y='Sales', data=df_StoreOpen, ax=axis1)
ax1.set(xlabel='State Holiday', ylabel='Sales')
ax2 = sns.barplot(x='SchoolHoliday', y='Sales', data=df_StoreOpen, ax=axis2)
ax2.set(xlabel='School Holiday', ylabel='Sales')
sns.plt.suptitle('Effect of Holidays on Sales')


#================== dealing with ourliers/cleaning ===========

### ============ identify outliers model (loop through many stores) =================


from collections import defaultdict
#A defaultdict works exactly like a normal dict, but it is 
#initialized with a function (“default factory”) that takes no arguments and 
#provides the default value for a nonexistent key.


def findOutliers(df):
    outliers = defaultdict(lambda: 0)
    for feature in df[['Sales','Customers']].keys():
        
        # TODO: Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(df[feature], 25.0)
        #print("Q1:", Q1)        
        # TODO: Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(df[feature], 75.0)
        #print("Q3:", Q3)                
        # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = step = 1.5 * (Q3 - Q1)
        #print("step:", step)        
        # Display the outliers
        print "Data points considered outliers for the feature '{}':".format(feature)
     
        outliers_df = df[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step))]
        for index in outliers_df.index.values:
            outliers[index] += 1
        #display(outliers_df)
        print("outliers:", outliers)       
    # OPTIONAL: Select the indices for data points you wish to remove
    outliers_list = [index for (index, count) in outliers.iteritems() if count > 1]
    print "Index of outliers for more than one feature: {} ".format(sorted(outliers_list))
    return outliers_list

stores = list(set(df['Store'].astype('int').values.tolist()))
# creating a list of lists forthe stores since .isin is expecting it
i=0
stores_list=[]
while i<len(stores):
  stores_list.append(stores[i:i+1])
  i+=1

#Create the list of the outliers for all stores
WholeOutlierList = []
problemList = []

for i in stores_list:
    print(i)
    try:
        WholeOutlierList.append(findOutliers(df.query('Sales > 0').loc[df['Store'].isin(i)]))
    except:
        problemList.append(i)


# Clean the dataframe from the outliers
OutliersToDrop = list(chain(*WholeOutlierList))
dfNO = df.drop(OutliersToDrop)


# seasonality 

df['DateYM'] = df['Date'].apply(lambda x: (str(x)[:7]))
dfNO['DateYM'] = dfNO['Date'].apply(lambda x: (str(x)[:7]))
# with outliers
averageSalesPerMonth = df.query('Sales > 0').groupby('DateYM')["Sales"].mean()
averageCustomersPerMonth = df.query('Sales > 0').groupby('DateYM')["Customers"].mean()
averageSalesPerCustomersPerMonth = df.query('Sales > 0').groupby('DateYM')["Sales"].sum()/df.query('Sales > 0').groupby('DateYM')["Customers"].sum()

# no outliers ()

averageSalesPerMonthNO = dfNO.query('Sales > 0').groupby('DateYM')["Sales"].mean()
averageCustomersPerMonthNO = dfNO.query('Sales > 0').groupby('DateYM')["Customers"].mean()
averageSalesPerCustomersPerMonthNO = dfNO.query('Sales > 0').groupby('DateYM')["Sales"].sum()/dfNO.query('Sales > 0').groupby('DateYM')["Customers"].sum()


# with outliers
fig, (axis1,axis2,axis3) = plt.subplots(3,1,sharex=True,figsize=(15,8))

# plot average sales per month
ax1 = averageSalesPerMonth.plot(legend=True,ax=axis1,marker='o',title="Average Sales")
ax1.set_xticks(range(len(averageSalesPerMonth)))
ax1.set_xticklabels(averageSalesPerMonth.index.tolist(), rotation=90)
# plot average customers per month
ax2 = averageCustomersPerMonth.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Average Customers Per Month")
# plot revenue per customer by month
ax3 = averageSalesPerCustomersPerMonth.plot(label='Average Sales Per Customer',legend=True,ax=axis3,marker='o',rot=90,colormap="terrain",title="Sales Per Customers by Month")
# No outliers
ax1 = averageSalesPerMonthNO.plot(legend=True,ax=axis1,marker='o',title="Average Sales",color='red')
ax2 = averageCustomersPerMonthNO.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Average Customers Per Month",color='red')
ax3 = averageSalesPerCustomersPerMonthNO.plot(label='Average Sales Per Customer',legend=True,ax=axis3,marker='o',rot=90,colormap="terrain",title="Sales Per Customers by Month",color='red')



#one hot encoding ======= main dataset ========
# one shot encoding for the new features

#hot shot encoding (leave out the 'DateDayOfYear' and consider the 'DateWeekOfYear')
df_dum = pd.get_dummies(df[['DayOfWeek','DateWeekOfYear','DateMonth','DateYear',
'Promo','StateHoliday','SchoolHoliday','DateDayOfMonth']])

#  merging the datasets
df = pd.DataFrame(df[['Set','Date','Store','Open','Sales','Customers']]).join(df_dum)


#one hot encoding ======= store dataset ========
#========== hot shot encoding for df_stores ========

# delete non neseccary fields
df_store.drop(['CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek',
'Promo2SinceYear', 'Promo2_0'],inplace=True,axis=1,errors='ignore')



df_storeDum = pd.get_dummies(df_store.ix[:,1:len(df_store)])
df_store = pd.DataFrame(df_store['Store']).join(df_storeDum)

#  merging the datasets
df = pd.merge(df, df_store, how='left', on=['Store'])



# Sales/customers and competition 

# fill NaN values
df["CompetitionDistance"].fillna(df_store["CompetitionDistance"].median())

df_StoreOpen = df.query('Open == 1')


## distance and sales
ax = sns.lmplot(x="CompetitionDistance", y="Sales", data=df_StoreOpen)
ax.set(xlabel='Competition Distance', ylabel='Sales')
sns.plt.title('Relationship between Competition distance and Sales')


ax = sns.lmplot(x="CompetitionDistance", y="Customers", data=df_StoreOpen)
ax.set(xlabel='Competition Distance', ylabel='Customers')
sns.plt.title('Relationship between Competition distance and Customers')

## competitors related
df['competitionDaysInbusiness'] = df['Date'] - df['CompetitionOpenSince']
df['competitionDaysInbusiness'] = df['competitionDaysInbusiness'].fillna(0)

df['competitionDaysInbusiness'] = (df['competitionDaysInbusiness']/ np.timedelta64(1, 'D')).astype(int)

ax = sns.lmplot(x="competitionDaysInbusiness", y="Sales", data=df.query('Sales > 0 and competitionDaysInbusiness > 1'))
ax.set(xlabel='competition Days In Business', ylabel='Sales')
sns.plt.title('Competition Days in Business and Sales')

ax = sns.lmplot(x="competitionDaysInbusiness", y="Customers", data=df.query('Sales > 0 and competitionDaysInbusiness > 1'))
ax.set(xlabel='competition Days In Business', ylabel='Customers')
sns.plt.title('Competition Days in Business and Customers')





