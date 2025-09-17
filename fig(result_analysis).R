library(ggplot2)

# Assuming pca_2d already contains PC1, PC2, doc_id, kmeans_cluster, hc_cluster, dbscan_cluster

# Convert cluster columns to factors for coloring/shaping
pca_2d$kmeans_cluster <- as.factor(pca_2d$kmeans_cluster)
pca_2d$hc_cluster <- as.factor(pca_2d$hc_cluster)
pca_2d$dbscan_cluster <- as.factor(pca_2d$dbscan_cluster)

# Plot PCA with K-means clusters (colors), Hierarchical clusters (shapes), and DBSCAN (alpha for noise)
ggplot(pca_2d, aes(x = PC1, y = PC2)) +
  geom_point(aes(color = kmeans_cluster, shape = hc_cluster, alpha = dbscan_cluster), size = 3) +
  scale_color_manual(values = c("1" = "red", "2" = "blue", "3" = "green", "4" = "orange")) +
  scale_shape_manual(values = c("1" = 16, "2" = 17, "3" = 15, "4" = 18)) +
  scale_alpha_manual(values = c("0" = 0.3, "1" = 1)) +
  labs(title = "PCA-based Visualization of Clusters",
       x = "Principal Component 1 (PC1)",
       y = "Principal Component 2 (PC2)",
       color = "K-means Cluster",
       shape = "Hierarchical Cluster",
       alpha = "DBSCAN Cluster") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"))


#=========X============================X========================X========================X======================

# Required libraries
library(ggplot2)
library(FactoMineR)
library(factoextra)
library(dbscan)

# Assuming you have a TF-IDF matrix 'tfidf_matrix' (rows = reviews, cols = top 10 words)

# 1. Apply PCA
pca_res <- PCA(tf_idf_matrix_full, graph = FALSE)
pca_data <- data.frame(pca_res$ind$coord[,1:2])  # PC1 and PC2
colnames(pca_data) <- c("Principal_Component_1", "Principal_Component_2")

# 2. K-means clustering
set.seed(123)
kmeans_res <- kmeans(tf_idf_matrix_full, centers = 4, nstart = 25)
pca_data$KMeans_Cluster <- factor(kmeans_res$cluster)

# 3. Hierarchical clustering
dist_matrix <- dist(tfidf_matrix)
hc_res <- hclust(dist_matrix, method = "ward.D2")
hc_clusters <- cutree(hc_res, k = 4)
pca_data$Hierarchical_Cluster <- factor(hc_clusters)

# 4. DBSCAN clustering
dbscan_res <- dbscan(tfidf_matrix, eps = 0.8, minPts = 5)
pca_data$DBSCAN_Cluster <- factor(dbscan_res$cluster)

# 5. Melt the data for ggplot (to plot all three clusterings in one figure)
library(reshape2)
pca_melt <- melt(pca_data, id.vars = c("Principal_Component_1", "Principal_Component_2"),
                 variable.name = "Method", value.name = "Cluster")

# 6. Plotting
ggplot(pca_melt, aes(x = Principal_Component_1, y = Principal_Component_2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  facet_wrap(~Method) +
  labs(
    x = "Principal Component 1",
    y = "Principal Component 2",
    title = "Fig. 1. PCA Scatter Plot of Reviews with K-means, Hierarchical, and DBSCAN Clusters",
    color = "Cluster ID"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 10, face = "bold"),
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 7),
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7)
  )

