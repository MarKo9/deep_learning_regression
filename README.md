# deep_learning_regression

## Using deep learning (FFNN) to solve a sales prediction problem

Revenue prediction for the second largest drugstore chain in Germany with over 3,000 drug stores in 7 European countries. Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. Specifically the target is to predict 6 weeks of daily sales for 1,115 stores located across Germany. 

The libraries used in the project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import math
import timeit
import seaborn as sns
import matplotlib.patches as mpatches
from collections import defaultdict
from itertools import chain
from sklearn import decomposition
from sklearn import preprocessing

The whole implementation was operated with Tensorflow as the main framework and in Spyder IDE (Anaconda).


Files included:
processing.py: Includes the operations for processing the data, cleaning and feature engineering.
processingNoOutliers.py Includes the operations for processing the data, cleaning and feature engineering plus the operation for removing the outliers based on the Tukey's method.
model.py: Is the file with the FFNN implementation. It takes as input data the csv file that the "processing.py" exports. It also includes the code for the visualizations on the model performance.
EDA.py: Includes some processing and the code used for the EDA section in the project.


