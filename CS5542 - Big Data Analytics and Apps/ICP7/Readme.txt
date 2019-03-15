Things Learnt:

K-Means Clustering - (k defines number of clusters):
first convert the data in to k clusters using density distribution 
calculates centroid of each cluster and re-assigns the data close to the clusters and creates new clusters 
repeat it untill no new clusters are created.

Sample code: 
val kMeansModel=KMeans.train(tf,10,1000) // trains a model for clustering (k=10)
val WSSSE = kMeansModel.computeCost(tf) // creates 10 clusters using the data in tf

Expectation Maximization Clustering:
Clustering happens based on probability of the data belongs to that cluster.

Sample code: 
val gmm = new GaussianMixture().setK(10).run(tf)
val clusters=gmm.predict(tf)