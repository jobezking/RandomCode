import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

class SupervisedWorker:   
    def __init__(self, csv_file, header=False):
        self.csv_file = csv_file
        self.header = header

        # placeholders for data
        self.X_d = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def load_data_supervised(self):
        if self.header:
            df = pd.read_csv(self.csv_file)
        else:
            df = pd.read_csv(self.csv_file, header=None)
        self.X_d = df.iloc[:, :-1]
        self.Y = df.iloc[:, -1:]

#####
def main():
    pass

#####
if __name__ == "__main__":
    main()  