import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

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
        self.Y_pred = None

    def load_data_supervised(self):
        if self.header:
            df = pd.read_csv(self.csv_file)
        else:
            df = pd.read_csv(self.csv_file, header=None)
        self.X_d = df.iloc[:, :-1]
        self.Y = df.iloc[:, -1:]

    def min_max_normalize(self):
        normalizer = MinMaxScaler()
        self.X = pd.DataFrame(normalizer.fit_transform(self.X_d), columns=self.X_d.columns)

    def zscore_normalize(self):
        self.X = (self.X_d - self.X_d.mean()) / self.X_d.std()

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=random_state
        )

    def performKNN(self, k=3):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(self.X_train, self.Y_train.values.ravel())
        self.Y_pred = knn.predict(self.X_test)
        #accuracy = accuracy_score(self.Y_test, self.Y_pred)

    def performLinearRegression(self):
        lr = LinearRegression()
        lr.fit(self.X_train, self.Y_train)
        self.Y_pred = lr.predict(self.X_test)
        #mse = mean_squared_error(self.Y_test, self.Y_pred)
        #r2 = r2_score(self.Y_test, self.Y_pred)

    def plotKNN(self, output="Save"):
        cm = confusion_matrix(self.Y_test, self.Y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for KNN')
        if output == "Show":
            plt.show()
        elif output == "Both":
            plt.show()
            plt.savefig("KNN-Confusion-Matrix.png")
        else:
            plt.savefig("KNN-Confusion-Matrix.png")

    def plotLinearRegression(self, output="Save"):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.Y_test, self.Y_pred, alpha=0.7)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Linear Regression: Actual vs Predicted')
        plt.plot([self.Y_test.min(), self.Y_test.max()], [self.Y_test.min(), self.Y_test.max()], 'r--')
        if output == "Show":
            plt.show()
        elif output == "Both":
            plt.show()
            plt.savefig("Linear-Regression-Actual-vs-Predicted.png")
        else:
            plt.savefig("Linear-Regression-Actual-vs-Predicted.png")
#####
def main():
    pass

#####
if __name__ == "__main__":
    main()  