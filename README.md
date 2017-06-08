# deep_learning_regression

## Using Feedforward neural network to solve a Sales prediction problem

Revenue prediction for the second largest drugstore chain in Germany with over 3,000 drug stores in 7 European countries. Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. Specifically the target is to predict 6 weeks of daily sales for 1,115 stores located across Germany. 

### The libraries used in the project

import pandas as pd<br />
import numpy as np<br />
import matplotlib.pyplot as plt<br />
import tensorflow as tf<br />
from sklearn.preprocessing import StandardScaler<br />
import math<br />
import timeit<br />
import seaborn as sns<br />
import matplotlib.patches as mpatches<br />
from collections import defaultdict<br />
from itertools import chain<br />
from sklearn import decomposition<br />
from sklearn import preprocessing<br />
The whole implementation was operated with Tensorflow as the main framework and in Spyder IDE (Anaconda).<br />


### Files included:

**processing.py:** Includes the operations for processing the data, cleaning and feature engineering.<br />
**processingNoOutliers.py:** Includes the operations for processing the data, cleaning and feature engineering plus the operation for removing the outliers based on the Tukey's method.<br />
**model.py:** Is the file with the FFNN implementation. It takes as input data the csv file that the "processing.py" exports. It also includes the code for the visualizations on the model performance.<br />
**EDA.py:** Includes some processing and the code used for the EDA section in the project.<br />


