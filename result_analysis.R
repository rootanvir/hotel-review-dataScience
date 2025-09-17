# =======================
# 1. PCA for 2D visualization
# =======================
pca_res <- prcomp(tf_idf_matrix_top10 %>% select(-doc_id), scale. = TRUE)
pca_2d <- as.data.frame(pca_res$x[, 1:2])
pca_2d$doc_id <- tf_idf_matrix_top10$doc_id

# Add cluster assignments for K-means
pca_2d$kmeans_cluster <- kmeans_result$cluster

# Add cluster assignments for Hierarchical
pca_2d$hc_cluster <- clusters

# Add cluster assignments for DBSCAN
pca_2d$dbscan_cluster <- dbscan_result$cluster

# =======================
# 2. Print PCA + cluster assignments
# =======================
print(pca_2d)

# =======================
# 3. Number of points per cluster
# =======================
cat("\nK-means cluster sizes:\n")
print(table(kmeans_result$cluster))

cat("\nHierarchical cluster sizes:\n")
print(table(clusters))

cat("\nDBSCAN cluster sizes (0 = noise):\n")
print(table(dbscan_result$cluster))

# =======================
# 4. K-means centroids
# =======================
cat("\nK-means cluster centroids:\n")
print(kmeans_result$centers)

