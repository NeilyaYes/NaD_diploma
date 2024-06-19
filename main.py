# For data analysis
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# For data modelling
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class k_means:
    def __init__(self, path='', outliers_fraction=0.01):
        self.filepath = path
        self.df = pd.read_csv(self.filepath)
        self.outliers_fraction = outliers_fraction

    def pre_processing(self):
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['hours'] = self.df['timestamp'].dt.hour
        self.df['daylight'] = ((self.df['hours'] >= 7) & (self.df['hours'] <= 19)).astype(int)
        self.df['DayofWeek'] = self.df['timestamp'].dt.dayofweek
        self.df['WeekDay'] = (self.df['DayofWeek'] < 5).astype(int)
        self.df['time_epoch'] = (self.df['timestamp'].astype(np.int64) / 100000000000).astype(np.int64)

    def standarizing(self):
        data = self.df[['value', 'hours', 'daylight', 'DayofWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)
        return data

    def getDistanceByPoint(self, data, model):
        distance = pd.DataFrame()
        for i in range(len(data)):
            Xa = np.array(data.loc[i])
            Xb = model.cluster_centers_[model.labels_[i] - 1]
            distance.at[i, 0] = np.linalg.norm(Xa - Xb)
        return distance

    def training(self, data):
        n_clusters = range(1, min(6, len(data) + 1))  # Ensure the number of clusters does not exceed the number of samples
        K_means = [KMeans(n_clusters=i).fit(data) for i in n_clusters]
        return K_means[1], data  # Adjust the index to select an appropriate model

    def predict(self, K_means, data):
        self.df['cluster'] = K_means.predict(data)
        distance = self.getDistanceByPoint(data, K_means)
        number_of_outliers = int(self.outliers_fraction * len(distance))
        threshold = distance.nlargest(number_of_outliers, 0).min()
        self.df['anomaly_kmeans'] = (distance >= threshold).astype(int)

    def visualisation(self):
        fig, ax = plt.subplots()
        a = self.df.loc[self.df['anomaly_kmeans'] == 1, ['time_epoch', 'value']]
        ax.plot(self.df['time_epoch'], self.df['value'], color='grey', alpha=0.3, label='Recorded Data')
        ax.scatter(a['time_epoch'], a['value'], color='red', label='Anomalies')
        ax.set_xlabel('Time EPOCH')
        ax.set_ylabel('Value')
        ax.legend()
        ax.set_title('Anomaly Detection (K-means Algorithm)')
        fig.savefig('./static/image/kmeans_plot.png', dpi=150, transparent=True)

    def evaluate(self, data):
        predicted_labels = self.df['anomaly_kmeans']
        unique_labels = len(set(predicted_labels))
        if unique_labels > 1:
            silhouette = round(silhouette_score(data, predicted_labels), 3)
            calinski_harabasz = round(calinski_harabasz_score(data, predicted_labels), 3)
            davies_bouldin = round(davies_bouldin_score(data, predicted_labels), 3)
            evaluation_str = (f"KMeans Evaluation:\nSilhouette Score: {silhouette}\n"
                              f"Calinski-Harabasz Index: {calinski_harabasz}\n"
                              f"Davies-Bouldin Index: {davies_bouldin}")
        else:
            evaluation_str = "KMeans Evaluation: No anomalies detected."

        return {
            "evaluation": evaluation_str,
            "anomalies": self.df[['timestamp', 'value', 'anomaly_kmeans']].to_dict(orient='records')
        }



class gaussian:
    def __init__(self, path='', outliers_fraction=0.01, support_fraction=0.999):
        self.filepath = path
        self.df = pd.read_csv(self.filepath)
        self.contamination = outliers_fraction
        self.support_fraction = support_fraction

    def pre_processing(self):
        if self.df.empty:
            raise ValueError("The input data is empty.")

        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['hours'] = self.df['timestamp'].dt.hour
        self.df['daylight'] = ((self.df['hours'] >= 7) & (self.df['hours'] <= 19)).astype(int)
        self.df['DayofWeek'] = self.df['timestamp'].dt.dayofweek
        self.df['WeekDay'] = (self.df['DayofWeek'] < 5).astype(int)
        self.df['time_epoch'] = (self.df['timestamp'].astype(np.int64) / 1000000000).astype(np.int64)

    def predict(self):
        self.df['categories'] = self.df['WeekDay'] * 2 + self.df['daylight']
        all_categories = []

        for i in range(4):
            category_data = self.df.loc[self.df['categories'] == i, 'value']
            if len(category_data) < 2:
                print(f"Skipping category {i} due to insufficient samples.")
                continue

            envelope = EllipticEnvelope(contamination=self.contamination, support_fraction=self.support_fraction)
            X_train = category_data.values.reshape(-1, 1)
            try:
                envelope.fit(X_train)
            except ValueError:
                print(f"Skipping category {i} due to singular covariance matrix.")
                continue

            category_df = pd.DataFrame(category_data)
            category_df['deviation'] = envelope.decision_function(X_train)
            category_df['anomaly'] = envelope.predict(X_train)
            all_categories.append(category_df)

        if all_categories:
            df_class = pd.concat(all_categories)
            self.df = self.df.merge(df_class[['deviation', 'anomaly']], left_index=True, right_index=True, how='left')
            self.df['anomaly_gaussian'] = np.array(self.df['anomaly'] == -1).astype(int)
        else:
            self.df['anomaly_gaussian'] = 0  # No anomalies if no category had enough samples

    def visualisation(self):
        fig, ax = plt.subplots()
        a = self.df.loc[self.df['anomaly_gaussian'] == 1, ('time_epoch', 'value')]
        ax.set_xlabel('Time EPOCH')
        ax.set_ylabel('Value')
        ax.plot(self.df['time_epoch'], self.df['value'], color='grey', alpha=0.3, label='Recorded Data')
        ax.scatter(a['time_epoch'], a['value'], color='red', label='Anomalies')
        ax.set_title('Anomaly Detection (Gaussian Mixture)')
        ax.legend()
        fig.savefig('./static/image/gaussian.png', dpi=150, transparent=True)

    def evaluate(self, data):
        predicted_labels = self.df['anomaly_gaussian']
        unique_labels = len(set(predicted_labels))
        if unique_labels > 1:
            silhouette = round(silhouette_score(data, predicted_labels), 3)
            calinski_harabasz = round(calinski_harabasz_score(data, predicted_labels), 3)
            davies_bouldin = round(davies_bouldin_score(data, predicted_labels), 3)
            evaluation_str = (f"GaussianMixture Evaluation:\nSilhouette Score: {silhouette}\n"
                              f"Calinski-Harabasz Index: {calinski_harabasz}\n"
                              f"Davies-Bouldin Index: {davies_bouldin}")
        else:
            evaluation_str = "GaussianMixture Evaluation: No anomalies detected."

        return {
            "evaluation": evaluation_str,
            "anomalies": self.df[['timestamp', 'value', 'anomaly_gaussian']].to_dict(orient='records')
        }

class isolationForest:
    def __init__(self, path='', outliers_fraction=0.01):
        self.filepath = path
        self.df = pd.read_csv(self.filepath)
        self.outliers_fraction = outliers_fraction

    def pre_processing(self):
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['hours'] = self.df['timestamp'].dt.hour
        self.df['daylight'] = ((self.df['hours'] >= 7) & (self.df['hours'] <= 19)).astype(int)
        self.df['DayofWeek'] = self.df['timestamp'].dt.dayofweek
        self.df['WeekDay'] = (self.df['DayofWeek'] < 5).astype(int)
        self.df['time_epoch'] = (self.df['timestamp'].astype(np.int64) / 1000000000).astype(np.int64)

    def standarizing(self):
        data = self.df[['value', 'hours', 'daylight', 'DayofWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)
        return data

    def training(self, data):
        iso_model = IsolationForest(contamination=self.outliers_fraction, n_estimators=200, max_samples=256, random_state=42)
        iso_model.fit(data)
        return iso_model, data

    def predict(self, iso_model, data):
        if self.df['value'].nunique() == 1:
            self.df['anomaly_isolationForest'] = 0
        else:
            self.df['anomaly_isolationForest'] = pd.Series(iso_model.predict(data))
            self.df['anomaly_isolationForest'] = (self.df['anomaly_isolationForest'] == -1).astype(int)

    def visualisation(self):
        fig, ax = plt.subplots()
        a = self.df.loc[self.df['anomaly_isolationForest'] == 1, ['time_epoch', 'value']]
        ax.plot(self.df['time_epoch'], self.df['value'], color='grey', alpha=0.3, label='Recorded Data')
        ax.scatter(a['time_epoch'], a['value'], color='red', label='Anomalies')
        ax.set_xlabel('Time EPOCH')
        ax.set_ylabel('Value')
        ax.set_title('Anomaly Detection (Isolation Forest)')
        ax.legend()
        fig.savefig('./static/image/isoForest.png', dpi=150, transparent=True)

    def evaluate(self, data):
        predicted_labels = self.df['anomaly_isolationForest']
        unique_labels = len(set(predicted_labels))
        if unique_labels > 1:
            silhouette = round(silhouette_score(data, predicted_labels), 3)
            calinski_harabasz = round(calinski_harabasz_score(data, predicted_labels), 3)
            davies_bouldin = round(davies_bouldin_score(data, predicted_labels), 3)
            evaluation_str = (f"IsolationForest Evaluation:\nSilhouette Score: {silhouette}\n"
                              f"Calinski-Harabasz Index: {calinski_harabasz}\n"
                              f"Davies-Bouldin Index: {davies_bouldin}")
        else:
            evaluation_str = "IsolationForest Evaluation: No anomalies detected."

        return {
            "evaluation": evaluation_str,
            "anomalies": self.df[['timestamp', 'value', 'anomaly_isolationForest']].to_dict(orient='records')
        }

class unsup_knn:
    def __init__(self, path='', threshold=1.5):
        self.filepath = path
        self.df = pd.read_csv(self.filepath)
        self.threshold = threshold

    def pre_processing(self):
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['hours'] = self.df['timestamp'].dt.hour
        self.df['daylight'] = ((self.df['hours'] >= 7) & (self.df['hours'] <= 19)).astype(int)
        self.df['DayofWeek'] = self.df['timestamp'].dt.dayofweek
        self.df['WeekDay'] = (self.df['DayofWeek'] < 5).astype(int)
        self.df['time_epoch'] = (self.df['timestamp'].astype(np.int64) / 1000000000).astype(np.int64)

    def standarizing(self):
        data = self.df[['value', 'hours', 'daylight', 'DayofWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)
        return data

    def training(self, data):
        knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
        knn.fit(data)
        return knn, data

    def predict(self, knn, data):
        distance, indices = knn.kneighbors(data)
        distance = pd.DataFrame(distance)
        distance_mean = distance.mean(axis=1)
        self.df['anomaly_unsupKNN'] = (distance_mean > self.threshold).astype(int)

    def visualisation(self):
        outlier_index = np.where(self.df['anomaly_unsupKNN'] == 1)
        outlier_values = self.df.iloc[outlier_index]
        fig, ax = plt.subplots()
        ax.plot(self.df["time_epoch"], self.df["value"], color="grey", alpha=0.3, label='Recorded Data')
        ax.scatter(outlier_values["time_epoch"], outlier_values["value"], color="red", label='Anomalies')
        ax.set_xlabel('Time EPOCH')
        ax.set_ylabel('Value')
        ax.set_title('Anomaly Detection (Unsupervised KNN)')
        ax.legend()
        fig.savefig('./static/image/unsupKNN.png', dpi=150, transparent=True)

    def evaluate(self, data):
        predicted_labels = self.df['anomaly_unsupKNN']
        unique_labels = len(set(predicted_labels))
        if unique_labels > 1:
            silhouette = round(silhouette_score(data, predicted_labels), 3)
        else:
            silhouette = "N/A"
        calinski_harabasz = round(calinski_harabasz_score(data, predicted_labels), 3) if unique_labels > 1 else "N/A"
        davies_bouldin = round(davies_bouldin_score(data, predicted_labels), 3) if unique_labels > 1 else "N/A"
        return {
            "evaluation": f"UnsupervisedKNN Evaluation:\nSilhouette Score: {silhouette}\nCalinski-Harabasz Index: {calinski_harabasz}\nDavies-Bouldin Index: {davies_bouldin}",
            "anomalies": self.df[['timestamp', 'value', 'anomaly_unsupKNN']].to_dict(orient='records')
        }