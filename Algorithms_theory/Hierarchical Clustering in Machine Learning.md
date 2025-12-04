Hiearchical clustering is an usupervised learning methid used to group similar data points into clusters based on their distance or similarity. Instead of choosing the number of clusters in advance , it bui,lds a tree like structure called a dendrogram that shows how clusters merge or split at different levels. It helps identify natural groupings in data and is commonly used in pattern recognition customer segmentation , gene analysis and image grouping . 
## Dendrogram : 
---
is like a family tree for clusters. It shows how individual data points or groups of data merge together. The bottom shows each data point as its own group and as we move up, similar groups are combined.<br>
The lower the merge point , the more similar the groups are. It helps us see how things are grouped step by step. 
![[Pasted image 20251204135502.png]]
- At the bottom of the denrogram the points are all separate 
- as we move up . the closest points are merged into a single group 
- The lines connecting the points show how they are progressively merged based on similarity 
- The height at which they are connected shows how similar the points are to each other ; the shorter the line the more similar they are . 
## Type of Hierarchical clustering 
---
there are two main types of Hierarchical clustering : 
## 1 . hierarchical agglomerative clustering : 
---
also known as the bottom-up approach or hieararchical agglomerative clustering . Bottom up algorithms treat each data as a singleton cluster at the outset and then successively agglomerate pairs of clusters until all clusters have been merged into a single cluster that contains all the data. 
### workflow for Hierarchical agglomerative clustering : 
---
1. Start with individual points : each data point is its own cluster. or example if we have 5 data points we start with 5 clusters each containing just one data point.
2. Calculate distances between clusters :  Calculate the distance between every pair of clusters. Initially since each cluster has one point this is the distance between the two data points.
3. merge the closest clusters:  Identify the two clusters with the smallest distance and merge them into a single cluster.
4. Update distance matrix : After merging we now have one less cluster. Recalculate the distances between the new cluster and the remaining clusters.
5. Repeat steps 3 and 4 : Keep merging the closest clusters and updating the distance matrix until we have only one cluster left.
6. create a dendrogram : As the process continues we can visualize the merging of clusters using a tree-like diagram called a dendrogram. It shows the hierarchy of how clusters are merged.
## 2. Hiearchical Divise clustering : 
---
divisive clustering is also known as a top-down approach. Top-down clustering requires a method for splitting a cluster that contains the whole data and proceeds by splitting clusters recursively until individual data have been split into singleton clusters.
### workflow for hiearchical divisive clustering : 
---
1. start with all data points in one cluster : Treat the entire dataset as a single large cluster.
2. Split the cluster :  Divide the cluster into two smaller clusters. The division is typically done by finding the two most dissimilar points in the cluster and using them to separate the data into two parts.
3. Repeat the process : For each of the new clusters, repeat the splitting process: Choose the cluster with the most dissimilar points and split it again into two smaller clusters.
4. stop when each data point is in its own cluster : Continue this process until every data point is its own cluster or the stopping condition (such as a predefined number of clusters) is met.