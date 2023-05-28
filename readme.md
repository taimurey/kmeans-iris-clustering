# Iris Flower Clustering Project

This project demonstrates how to use K-means clustering to group Iris flowers based on their petal width and length. We only use these two features for simplicity and visualize the clusters in a 2D plot. The project also involves scaling of features and determining the optimal number of clusters using the Elbow method.

## Table of Contents

1. [Requirements](#requirements)
2. [Dataset](#dataset)
3. [Steps](#steps)
4. [Code](#code)
5. [Results](#results)
6. [Future Improvements](#future-improvements)

## Requirements <a name="requirements"></a>

- Python 3.x
- Libraries: sklearn, pandas, numpy, matplotlib

## Dataset <a name="dataset"></a>

The dataset used is the famous Iris flower dataset, which comes preloaded with sklearn's datasets module. The dataset contains 150 samples of iris flowers from three different species. There are four features for each sample: sepal length, sepal width, petal length, and petal width. For this project, we are only using the petal length and width features.

## Steps <a name="steps"></a>

1. Load the Iris dataset and create a pandas DataFrame with only the petal length and width features.

2. Scale the data using MinMaxScaler from sklearn.preprocessing. This is done because the K-means algorithm is distance based and can be biased towards higher dimensional features.

3. Apply K-means clustering with an initial guess for the number of clusters (e.g. 3). Add the resulting cluster assignments to the DataFrame.

4. Visualize the clusters in a scatter plot, where each cluster is a different color and the cluster centroids are marked with a star.

5. Finally, to find the optimal number of clusters, we use the Elbow method. This involves running K-means with a range of different K values and plotting the sum of squared errors (SSE) for each K. The 'elbow' of the plot, where the rate of decrease of SSE slows significantly, suggests the optimal K value.

## Code <a name="code"></a>

Here is the Python code for the project:

```python

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)
y_predicted = kmeans.fit_predict(df)
df['cluster'] = y_predicted

# Visualize clusters
colors = ['green', 'red', 'black']
for i in range(3):
    plt.scatter(df[df.cluster == i][df.columns[0]], df[df.cluster == i][df.columns[1]], color=colors[i])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()
plt.show()

```

## Results

The scatter plot shows the three clusters formed from the petal length and width features. The Elbow plot suggests that the optimal number of clusters is around 3, which corresponds to the number of species in the Iris dataset.

---

## Future Improvements

Future improvements could include using all four features in the Iris dataset and visualizing the clusters in a 3D plot, or experimenting with different clustering algorithms and comparing their results.

---
