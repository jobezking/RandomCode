import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix

class UnsupervisedWorker:   
    def __init__(self, csv_file, header=False):
        self.csv_file = csv_file
        self.header = header

        # placeholders for data
        self.X_df = None
        self.labels = None
        self.centroids = None

    def load_data_unsupervised(self):
       if self.header:
          self.X_df = pd.read_csv(self.csv_file)
       else:
           self.X_df = pd.read_csv(self.csv_file, header=None)

    def zscore_normalize(self):
        self.X_df = (self.X_df - self.X_df.mean()) / self.X_df.std()
    
    def runKMeans(self, K=3):
        kmeans = KMeans(K, random_state=42, n_init='auto')
        kmeans.fit(self.X_df.values)
        self.labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_
    
    def plot_KMeans(self, K, output="Save"):
        X = self.X_df.values
        if X.shape[1] != 2:
            raise ValueError("Data must be 2-dimensional for plotting.")
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=self.labels, palette='viridis', s=100)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
        plt.title('K-Means Clustering')
        plt.legend()
        if output == "Show":
            plt.show()
        elif output == "Both":
            plt.show()
            plt.savefig(f"K-means-{K}.png")
        else:
            plt.savefig(f"K-means-{K}.png")


################
def main():
    pass

#####
if __name__ == "__main__":
    main()  
