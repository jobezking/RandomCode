# do_kNN.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

class BasicKNN:
    def __init__(self, csv_file, supervised=True, header=False):
        self.csv_file = csv_file
        self.supervised = supervised
        self.header = header

        if self.supervised: 
            self.load_data = self.load_data_supervised()
        
        # placeholders for data
        self.X_d = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.distances = None
        self.predictions = []
###########################################################
#   def load_data_unsupervised(self):  # will not be used in this example but done for future use
#       if self.header:
#           df = pd.read_csv(self.csv_file)
#       else:
#           df = pd.read_csv(self.csv_file, header=None)
#       X_d = df.astype(float)
#       return self.normalize_cluster_data(X_d) 
#
#  def normalize_cluster_data(self, X_d):
#      return (X_d - X_d.mean()) / X_d.std()
##########################################################
    def load_data_supervised(self):
        if self.header:
            df = pd.read_csv(self.csv_file)
        else:
            df = pd.read_csv(self.csv_file, header=None)
        self.X_d = df.iloc[:, :-1].astype(float)
        self.Y = df.iloc[:, -1:].astype(float) 

    def normalize_distance_data(self):
        normalizer = MinMaxScaler()
        self.X = pd.DataFrame(normalizer.fit_transform(self.X_d), columns=self.X_d.columns)

    def data_split_test_train(self, train_split_val="70%", seed=50):
        Y, X = self.load_data()
        train_split = float(train_split_val.strip('%')) / 100
        total_rows = X.shape[0]
        train_size = int(train_split * total_rows)

        rng = np.random.default_rng(seed)
        sample_idx = rng.permutation(X.shape[0])

        self.X_train, self.X_test = X[sample_idx[0:train_size]], X[sample_idx[train_size:total_rows]]
        self.Y_train, self.Y_test = Y[sample_idx[0:train_size]], Y[sample_idx[train_size:total_rows]]

    def calculate_distances(self):
        self.distances = np.sqrt(np.sum((self.X_test[:, None, :] - self.X_train[None, :, :]) ** 2, axis=2))

    def assign_knn(self, k):
        neighbor_indices = np.argsort(self.distances, axis=1)[:, :k]

        for indices in neighbor_indices:
            labels = self.Y_train[indices]
            values, counts = np.unique(labels, return_counts=True)
            predicted_label = values[np.argmax(counts)]
            self.predictions.append(predicted_label)

    def visualize(self, k, output="Save"):
        y_pred = np.array(self.predictions)
        y_true = np.array(self.Y_test)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        labels = np.unique(y_true)

        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix (k={k}), Accuracy={acc:.2f}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        if output == "Show":
            plt.show()
        elif output == "Both":
            plt.show()
            plt.savefig(f"confusion_matrix-{k}.png")
        else:
            plt.savefig(f"confusion_matrix-{k}.png")


def main():
    
    # Instantiate the class and create classifier object using data file, split percentage, random seed
    input_data_file = "submissions/wdbc.data.mb.csv"
    kNN = BasicKNN(csv_file = input_data_file, supervised=True, header=False)

    # Normalize data for distance calculations for question 1
    kNN.normalize_distance_data()           
    # Prepare training and testing data sets for question 2
    kNN.data_split_test_train(train_split_val="70%", seed=42)
    
    # calculate all distances between test and train data points for question 3
    kNN.calculate_distances()

    # Evaluate different values for k
    k_values = [1, 3, 5, 7, 9] # question 5
    for k in k_values:      # question 5
        kNN.assign_knn(k)  # assignment module where it will find k-Nearest Neighbors to classify samples with unknown class assignment question 4
        kNN.visualize(k, output="Save") # question 6

####
if __name__ == "__main__":
    main()

