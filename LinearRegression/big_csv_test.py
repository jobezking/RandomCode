import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#
#pd.set_option('future.no_silent_downcasting', True)
#modeldata = pd.read_csv("/home/jeking/big_csv/kaggle/hospital_prices.csv",low_memory=False) #load data
warnings.filterwarnings("ignore")
try:
    #modeldata = pd.read_csv("/home/jeking/big_csv/kaggle/used_cars_data.csv",low_memory=False) #load data
    modeldata = pd.read_csv("/home/jeking/big_csv/kaggle/large_stress_test.csv",low_memory=False) #load data
    print("DataFrame loaded successfully!")
    print(f"DataFrame size: {modeldata.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
except MemoryError as e:
    print(f"MemoryError: Failed to load the entire DataFrame into memory. Error: {e}")
#
print(modeldata.head())
print(modeldata.tail())
