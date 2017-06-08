import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import math
import timeit
import seaborn as sns
import matplotlib.patches as mpatches


#from V_04_mlnd_capstone_NN_regressionInFuntion import trainNN

def rmspeM(y_hat, y):
    rmspe = np.sqrt(np.mean((((y-y_hat))/y)**2))
    return rmspe
# ==================== The NN funtion ==================    


epochCostList = []
def trainNN(train_x,train_y,test_x,test_y,epoch_n):
    features =  train_x.shape[1]     
    # st here the layers parameters later (line 128, Layers = )   
    with tf.name_scope("IO"):
        inputs = tf.placeholder(tf.float32, [None, train_x.shape[1]], name="X")
        outputs = tf.placeholder(tf.float32, [None, 1], name="Yhat") # change names
    
    with tf.name_scope("LAYER"):
        # network architecture, "features" start with the input features #
        Layers = [features, 20, 22,8, 1]# gives 1.3
        
        h1   = tf.Variable(tf.random_normal([Layers[0], Layers[1]], 0, 0.1, dtype=tf.float32, seed=0), name="h1")
        h2   = tf.Variable(tf.random_normal([Layers[1], Layers[2]], 0, 0.1, dtype=tf.float32, seed=0), name="h2")
        h3   = tf.Variable(tf.random_normal([Layers[2], Layers[3]], 0, 0.1, dtype=tf.float32, seed=0), name="h3")
        hout = tf.Variable(tf.random_normal([Layers[3], Layers[4]], 0, 0.1, dtype=tf.float32, seed=0), name="hout")
    
        b1   = tf.Variable(tf.random_normal([Layers[1]], 0, 0.1, dtype=tf.float32, seed=0 ), name="b1" )
        b2   = tf.Variable(tf.random_normal([Layers[2]], 0, 0.1, dtype=tf.float32, seed=0 ), name="b2" )
        b3   = tf.Variable(tf.random_normal([Layers[3]], 0, 0.1, dtype=tf.float32, seed=0 ), name="b3" )
        bout = tf.Variable(tf.random_normal([Layers[4]], 0, 0.1, dtype=tf.float32, seed=0 ), name="bout" )
           
      
    # The layer operations
        
    def model( inputs, layers ):
        [h1, b1, h2, b2,h3, b3, hout, bout] = layers
        y1 = tf.add( tf.matmul(inputs, h1), b1 )
        y1 = tf.nn.sigmoid( y1 )
           
        y2 = tf.add( tf.matmul(y1, h2), b2 )
        y2 = tf.nn.sigmoid( y2 )
    
        y3 = tf.add( tf.matmul(y2, h3), b3 )
        y3 = tf.nn.sigmoid( y3 )

        yret  = tf.matmul(y3, hout) + bout 
        return yret
           
    with tf.name_scope("train"):
        learning_rate = 0.7
        yout = model( inputs, [h1, b1, h2, b2,h3, b3, hout, bout] )
        
        cost_op = tf.reduce_mean( tf.pow( yout - outputs, 2 ))
        train_op = tf.train.AdagradOptimizer( learning_rate=learning_rate ).minimize( cost_op )

    # define variables/constants that control the training
    epoch = 0
    max_epochs = epoch_n 
    
    print( "Beginning Training" )
    
    sess = tf.Session() # Create TensorFlow session
    with sess.as_default():
        
        # tensorboard
        #writer = tf.summary.FileWriter("/Users/mariosk/tf_test")
        #writer.add_graph(sess.graph)
        
        # initialize the variables
        init = tf.global_variables_initializer()
        sess.run(init)
        
        costs = []
        epochs= []

        while True:
            # Do the training
            sess.run( train_op, feed_dict={inputs: train_x, outputs: train_y} )
                
            # Update the user every 1000 epochs
            if epoch % 100==0:
                cost = sess.run(cost_op, feed_dict={inputs: train_x, outputs: train_y})
                costs.append( cost )
                epochs.append( epoch )
                epochCostList.append([epochs,costs])
                
                print( "Epoch: %d - Error: %.4f" %(epoch, cost) )
                
                # time to stop?
                if epoch > max_epochs :
                    # or abs(last_cost - cost) < tolerance:
                    print( "STOP!" )
                    break
                
            epoch += 1
        
        # compute the predicted output for test_x
        pred_y = sess.run(yout, feed_dict={inputs: test_x, outputs: test_y} )
        
        return pred_y

# no outlier data (Can be exported from the "processingNoOutliers.py" file)
#data_train = pd.read_csv("train_dumNO.csv",nrows = None)
#data_test = pd.read_csv("test_dumNO.csv",nrows = None)
# With outlier data

# With outlier data
data_train = pd.read_csv("train_dum.csv",nrows = None)
data_test = pd.read_csv("test_dum.csv",nrows = None)


del data_train["Unnamed: 0"]
del data_test["Unnamed: 0"]
#data_store = pd.read_csv("store.csv")

# remove some features that did not effect the model positively
toDelete = ['StateHoliday_0','StateHoliday_a','StateHoliday_b','StateHoliday_c',
            'competitionDaysInbusiness','promoActiveDays']
      
data_train.drop(toDelete,axis=1, inplace=True,errors='ignore')
data_test.drop(toDelete,axis=1, inplace=True,errors='ignore')

# delete the points were the stores were open but with no sales
closed_store_data = data_test["Id"][data_test["Open"] == 0].values
# remove date for closed stores
data_test = data_test[data_test["Open"] != 0]
data_train = data_train[data_train["Open"] != 0]

data_test = data_test.fillna(0)
data_train = data_train.fillna(0)
# Make it shorter
data_test = data_test.query('Store==879') # 879
#data_test.shape

# Make it shorter
data_train = data_train.query('Store==879') # 879
#data_train.shape

# create dictionaries with the stores data
train_stores = dict(list(data_train.groupby('Store')))
test_stores = dict(list(data_test.groupby('Store')))


####################### calling the NN

# timing the training
start_time = timeit.default_timer()
result = pd.Series()
resultList = []
totalResults = []

# can test various values fort the epoch variable
epochs = [600]
for epoch_n in epochs:
    globals()['result%s' % (epoch_n)]   =   []
    for i in test_stores:  
        store = train_stores[i]
        X_train = store.drop(["Sales", "Store","Customers"],axis=1)
        Y_train = store["Sales"]
        X_test  = test_stores[i].copy() 
        
        # create the Y_test, delete 'sales' from X_test set
        Y_test  = X_test['Sales']     
        store_ind = X_test["Id"]
        X_test.drop(["Sales","Id","Store","Customers"], axis=1,inplace=True)
        
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())
    
        # add scaling
        scx = StandardScaler()
        scy = StandardScaler()
        X_train = scx.fit_transform(X_train) 
        X_test = scx.fit_transform(X_test)
        Y_train = scy.fit_transform(Y_train)
        Y_train = Y_train.reshape(-1, 1)
        Y_test = scy.fit_transform(Y_test)
        Y_test = Y_test.reshape(-1, 1)
    
        # calling the NN    
        print(i)
        Y_pred = trainNN(X_train, Y_train,X_test,Y_train,epoch_n)

        #revercing the scale to get the results
        Y_pred = scy.inverse_transform(Y_pred)
        # reshape back to 1D for the next step
        Y_pred = Y_pred.reshape(Y_pred.shape[0])
        
        globals()['result%s' % (epoch_n)].append(pd.Series(Y_pred, index=store_ind))
        #result = result.append(pd.Series(Y_pred, index=store_ind))
        total = pd.concat(globals()['result%s' % (epoch_n)])

    # saving predictions 
    totalResults.append(total)
    # get the timing result
    elapsed = timeit.default_timer() - start_time
    
# preparing the Y_test data for the cost calculation
Y_testWhole =  data_test['Sales'].values.reshape(data_test['Sales'].shape[0])
Y_testWhole = pd.DataFrame(Y_testWhole)
Y_testWhole.index = data_test["Id"]
Y_testWholeOr = math.e**Y_testWhole[0]

# create the list with the 
rmspeEpochList = []
e=0
for i in totalResults:
    wholePredOr = math.e**i
    wholePredOr=pd.DataFrame(wholePredOr)
    wholePredOr.sort_index(inplace=True)
    rmspeData = rmspeM(wholePredOr[0], Y_testWholeOr)  
    rmspeEpochList.append([epochs[e],rmspeData])
    e +=1   
    
# ploting RMSPE VS EPOCHS and cost (use only in when many values are used for 
    # the epoch hyperparameter)
# prepare the data sets
"""   
df_rmspe = pd.DataFrame(rmspeEpochList,columns=["epochs","rmspe"])
epochCost = pd.DataFrame(epochCostList[30][1],columns=["Cost"])
epochCost["epochs"]  = epochCostList[30][0]

ax = sns.pointplot(y="rmspe", x="epochs",data=df_rmspe, dodge=True,color="red")
ax = sns.pointplot(y="Cost", x="epochs",data=epochCost.query('epochs >200'),color="blue")
ax.set(ylabel='RMSPE', xlabel='Epoch')
sns.plt.title('Epochs Vs RMSPE and Cost (Initial model)')
rmspe = mpatches.Patch(color='red', label='RMSPE')
Cost = mpatches.Patch(color='blue', label='Cost')
plt.legend(handles=[rmspe,Cost])
plt.show()

"""


# plot the actual vs prediction 
rmspe = rmspeM(wholePredOr[0], Y_testWholeOr)
fig = plt.figure()
plt.plot(wholePredOr, label='Prediction')
plt.plot(Y_testWholeOr, label='Actual Sales')
fig.suptitle('Prediction Vs Actual Sales (6 weeks interval(Original term))')
plt.xlabel('Id')
plt.ylabel('Sales')
plt.legend(bbox_to_anchor=(1, 1))
plt.text(12000, 7000,'rmspe = %f' % rmspe, ha='center', va='center',bbox={'facecolor':'grey', 'alpha':0.5, 'pad':10})
plt.show()

# plot the residuals
fig = plt.figure()
plt.scatter( Y_testWholeOr, Y_testWholeOr - wholePredOr[0],alpha=0.3)
plt.axhline(0, color='blue')
plt.xlabel( "Actual" )
plt.ylabel( "Actual - Predicted" )
plt.title( "Residuals" )
plt.show()



