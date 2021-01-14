#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import pickle
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score


df = pd.read_csv('protein.csv')


id_col = df['ID']


# Data preperation
print('\n======== DATA PREPERATION ========\n\n')

df = df.drop(labels=['ID'], axis=1)

class OutlierClipper:
    def __init__(self, features, threshold = 1.5):
        self.threshold = threshold
        self._features = features
        self._feature_map = {}

    def fit(self, X, **kwargs):
        df = X[self._features]
        features = list(df.columns)
        for feature in features:
            # f_q1 = df[feature].quantile(0.25)
            # f_q3 = df[feature].quantile(0.75)
            # f_iqr = f_q3 - f_q1

            mean = X[feature].mean()
            std = X[feature].std(ddof=0)
            
            self._feature_map[feature] = (mean - (self.threshold * std), mean + (self.threshold * std))
        return self
    
    def count_outliers(self, data):
        for column in data.columns:
            too_low = sum(data[column] < self._feature_map[column][0])
            too_high = sum(data[column] > self._feature_map[column][1])
            print('Column {} has {} high values and {} low values. overall: {}'.format(column, too_high, too_low, too_high + too_low))

    def transform(self, data):
        data_copy = data.copy()
        for feature in self._feature_map.keys():
            data_copy[feature] = data_copy[feature].clip(lower=self._feature_map[feature][0],
                                                         upper=self._feature_map[feature][1])
        return data_copy

    def fit_transform(self, X, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)

print('\n\n --------- OUTLIER CLIPPING ---------- \n\n')
outlier_clipper = OutlierClipper(df.columns, threshold=1)
outlier_clipper.fit(df)
df = outlier_clipper.transform(df)
print(df.describe())


class Imputer:
    def __init__(self):
        self.iterative_imputer = IterativeImputer(initial_strategy='mean')

    def fit(self, X, **kargs):
        self.iterative_imputer.fit(X)
        # print(X.shape)
        return self

    def transform(self, X):
        columns = X.columns
        X = self.iterative_imputer.transform(X)
        X = pd.DataFrame(X, columns=columns)
        return X

    def fit_transform(self, X, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)

print('\n\n --------- DATA IMPUTATION ---------- \n\n')
imputer = Imputer()
imputer.fit(df)
df = imputer.transform(df)
print(df.isna().sum())

# Model evaluation
print('\n======== MODEL SELECTION ========\n\n')

print('\n\n --------- MODEL FITTING ---------- \n\n')

kmeans = KMeans(n_clusters=5, max_iter=500, n_init=30)

spectral = SpectralClustering(n_clusters=5,
        n_components=6,
        gamma=100,
        assign_labels="discretize")

gmm = GaussianMixture(n_components=5)

kmeans.fit(df)
spectral.fit(df)
gmm.fit(df)


## Scores
print('\n\n --------- MODEL SCORES ---------- \n\n')
print('davies_boulding score:')
print(f'KMEANS davies_boulding: {davies_bouldin_score(df, kmeans.labels_)}')
print(f'SPECTRAL davies_boulding: {davies_bouldin_score(df, spectral.labels_)}')
print(f'GNN davies_boulding: {davies_bouldin_score(df, gmm.predict(df))}')

print('\n\n silhouette_score:')
print(f'KMEANS silhouette_score: {silhouette_score(df, kmeans.labels_)}')
print(f'SPECTRAL silhouette_score: {silhouette_score(df, spectral.labels_)}')
print(f'GNN silhouette_score: {silhouette_score(df, gmm.predict(df))}')

## Select the best model

selected_model = kmeans

# Select most useful features
print('\n======== SELECT 5 BEST FEATURES ========\n\n')
def print_score(features):
    k = KMeans(n_clusters=5, max_iter=500, n_init=30)

    features = [f'protein_{i}' for i in features]


    k_fitted = k.fit(df[features])

    y = k_fitted.predict(df[features])

    print(f'davies_bouldin_score: {davies_bouldin_score(df, y)}')
    print(f'silhouette_score: {silhouette_score(df, y)}')


## By single feature score reduction
print('\n-------- FEATURE REDUCTION METHOD --------\n')
feature_scores = []
kmeans = KMeans(n_clusters=5, max_iter=500, n_init=30)
kmeans_temp = kmeans.fit(df)
for i in range(10):
    column = f'protein_{i+1}'
    df_temp = df.drop([column], axis=1)
    kmeans = KMeans(n_clusters=5, max_iter=500, n_init=30, random_state=0)
    kmeans_temp = kmeans.fit(df_temp)
    feature_scores.append(silhouette_score(df, kmeans_temp.labels_))
    
importance = pd.DataFrame(feature_scores)
importance.index = np.arange(1, len(importance) + 1)
best_5 = importance[0].nlargest(5).index.to_numpy()
print(best_5)

print_score(best_5)

## Using Random Forest

### We use a random forest classifier on the results of the kmeans model, and use the importance array
print('\n-------- RANDOM FOREST METHOD --------\n')
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(df, selected_model.predict(df))
importance = pd.DataFrame(clf.feature_importances_)
importance.index = np.arange(1, len(importance) + 1)
best_5 = importance[0].nlargest(5).index.to_numpy()
print(best_5)

print_score(best_5)


## Using PCA

### We use pca to reduce dimension to 5, and then use the importance matrix to rank the features
print('\n-------- PCA METHOD --------\n')
from sklearn.decomposition import PCA

pca = PCA(n_components=5)

df_hat = pca.fit_transform(df)

importance = pd.DataFrame(abs( pca.components_ ))

f = np.ndarray(shape=(5,1), dtype=np.int8)
for i in range(5):
    best_feat = importance.T[i].argmax()
    importance[best_feat] = 0
    f[i] = best_feat

f = f + 1
best_5 = f.flatten()
print(best_5)

print_score(best_5)


# Selected 5 best features

### We are using the RandomForest feature importance method

selected_features = [f'protein_{i}' for i in [4, 5, 6, 8, 9]]

kmeans = KMeans(n_clusters=5, max_iter=500, n_init=30)
selected_model = kmeans.fit(df[selected_features])


## Save results

result = pd.DataFrame()
result['ID'] = id_col
result['y'] = selected_model.labels_

result.to_csv(f'clusters.csv', index=False)


# Visualizations

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE


def reduce_dims_tsne_2d(X, y):
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
        
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    df_subset['y'] = y
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 5),
        data=df_subset,
        legend="full"
    )
    
def reduce_dims_tsne_3d(X, y):
    tsne = TSNE(n_components=3, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    
    df_subset = pd.DataFrame()
    df_subset['tsne-3d-one'] = tsne_results[:,0]
    df_subset['tsne-3d-two'] = tsne_results[:,1]
    df_subset['tsne-3d-three'] = tsne_results[:,2]
    df_subset['y'] = y
    
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df_subset["tsne-3d-one"], 
        ys=df_subset["tsne-3d-two"], 
        zs=df_subset["tsne-3d-three"], 
        c=df_subset["y"], 
        cmap='tab10'
    )
    
reduce_dims_tsne_2d(df, selected_model.labels_)
reduce_dims_tsne_3d(df, kmeans.labels_)


# Mutations Characteristics
print('\n\n========= MUTATION CHARACTERISTICS ========= \n\n')
## Mutation Prevelence


df_labeled = pd.DataFrame(df)
df_labeled['y'] = kmeans.labels_
print('--------- MUTATION PREVELENCE ---------')
print(df_labeled.y.value_counts(normalize=True) * 100)


## Mutation Centroids

from sklearn.neighbors import NearestCentroid

centroids_clf = NearestCentroid()
centroids_clf.fit(df, selected_model.predict(df[selected_features]))
centroids = centroids_clf.centroids_
print('--------- CLUSTER CENTROIDS ---------')
print(centroids)



## Nearest neighbour

from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5).fit(centroids)
distances, indices = nbrs.kneighbors(centroids)
print('--------- NEAREST NEIGHBOURS ---------')
print(distances)
