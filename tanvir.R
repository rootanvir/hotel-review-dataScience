library(readr)
library(tokenizers)
library(dplyr)
library(purrr)
library(tidytext)
library(tidyr)
library(tibble)
library(ggplot2)
library(FactoMineR)
library(factoextra)
library(dbscan)

# =======================
# Load data
# =======================

data <- read_csv("D:/data science final project/archive/7282_1_2k.csv")

# =======================
# Tokenization & Stop word
# =======================
tokenization_no_stop <- data %>%
  mutate(tokens = map(reviews.text, ~ {
    words <- tokenize_words(.x)[[1]]
    words <- tolower(words)
    words <- words[!words %in% get_stopwords(language = "en")$word]
    words
  })) %>%
  select(tokens)

# =======================
# Word frequencies
# =======================
word_freq <- tokenization_no_stop %>%
  unnest(tokens) %>%
  count(tokens, name = "frequency", sort = TRUE) %>%
  rename(word = tokens)

# =======================
# Top 10 words
# =======================
top_words <- word_freq %>%
  slice_max(frequency, n = 10) %>%
  pull(word)

# =======================
# TF
# =======================
tf_data <- tokenization_no_stop %>%
  mutate(doc_id = row_number()) %>%
  unnest(tokens) %>%
  count(doc_id, tokens, name = "term_count") %>%
  group_by(doc_id) %>%
  mutate(tf = term_count / sum(term_count)) %>%
  ungroup() %>%
  rename(word = tokens)

# =======================
# IDF
# =======================
total_docs <- nrow(tokenization_no_stop)
idf_data <- tokenization_no_stop %>%
  mutate(doc_id = row_number()) %>%
  unnest(tokens) %>%
  distinct(doc_id, tokens) %>%
  count(tokens, name = "doc_freq") %>%
  mutate(idf = log(total_docs / doc_freq)) %>%
  rename(word = tokens)

# =======================
# Full TF-IDF table
# =======================
tf_idf_full <- tf_data %>%
  left_join(idf_data, by = "word") %>%
  mutate(tf_idf = tf * idf) %>%
  left_join(word_freq, by = "word")  # add frequency column

# Full TF-IDF matrix (data table in R)
tf_idf_matrix_full <- tf_idf_full %>%
  select(doc_id, word, tf_idf) %>%
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%
  as_tibble()

# =======================
# Top 10 TF-IDF table
# =======================
tf_idf_top10 <- tf_idf_full %>%
  filter(word %in% top_words)

# Top 10 TF-IDF matrix (data table in R)
tf_idf_matrix_top10 <- tf_idf_top10 %>%
  select(doc_id, word, tf_idf) %>%
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%
  as_tibble()

# =======================
# K-means clustering on top 10 words
# =======================
set.seed(123)
k <- 4
kmeans_result <- kmeans(as.matrix(tf_idf_matrix_top10 %>% select(-doc_id)), centers = k, nstart = 25)

# Visualization
fviz_cluster(kmeans_result,
             data = as.matrix(tf_idf_matrix_top10 %>% select(-doc_id)),
             geom = "point",
             ellipse.type = "convex",
             repel = TRUE,
             show.clust.cent = TRUE) +
  ggtitle("K-means Clustering of Reviews (Top 10 Words)")




# =======================
# Hierarchical Clustering (Top 10 Words)
# =======================

# Convert TF-IDF top 10 table to numeric matrix (remove doc_id)
tf_idf_mat <- as.matrix(tf_idf_matrix_top10 %>% select(-doc_id))

# Compute distance matrix using Euclidean distance
dist_matrix <- dist(tf_idf_mat, method = "euclidean")

# Perform hierarchical clustering using Ward's method
hc <- hclust(dist_matrix, method = "ward.D2")

# Plot dendrogram
plot(hc, labels = FALSE, hang = -1, main = "Hierarchical Clustering Dendrogram (Top 10 Words)")

# Cut tree into k clusters
k <- 4
clusters <- cutree(hc, k = k)

# Add cluster assignments to the top 10 TF-IDF table
tf_idf_matrix_top10_clustered <- tf_idf_matrix_top10 %>%
  mutate(cluster = clusters)

#====X=============X=====================X==================X=======================X=====




#===============================
#            DBSCAN
#===============================

# =======================
# Prepare TF-IDF matrix
# =======================
tf_idf_mat <- as.matrix(tf_idf_matrix_top10 %>% select(-doc_id))

# =======================
# DBSCAN clustering
# =======================
# eps = neighborhood radius, minPts = minimum points in a cluster
dbscan_result <- dbscan(tf_idf_mat, eps = 0.8, minPts = 5)

# Cluster assignments
tf_idf_matrix_top10_dbscan <- tf_idf_matrix_top10 %>%
  mutate(cluster = dbscan_result$cluster)

# =======================
# Visualize DBSCAN clusters (2D using PCA)
# =======================
fviz_cluster(list(data = tf_idf_mat, cluster = dbscan_result$cluster),
             geom = "point",
             ellipse = FALSE,
             repel = TRUE,
             show.clust.cent = TRUE) +
  ggtitle("DBSCAN Clustering of Reviews (Top 10 Words)")

#=========X=================X======================X========================





#============================================================
#                             PCA
#================================================================

# Prepare TF-IDF matrix (remove doc_id)
tf_idf_mat <- as.matrix(tf_idf_matrix_top10 %>% select(-doc_id))

# Apply PCA
pca_result <- prcomp(tf_idf_mat, scale. = TRUE)

# Take first 2 principal components
pca_2d <- as.data.frame(pca_result$x[, 1:2])
pca_2d$doc_id <- tf_idf_matrix_top10$doc_id  # optional: keep doc IDs

# Optional: visualize the points in 2D
ggplot(pca_2d, aes(x = PC1, y = PC2)) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "PCA 2D Projection of Top 10 TF-IDF Features",
       x = "PC1", y = "PC2") +
  theme_minimal()
#==========X===============X===================X======================



#====================================================================
#                 PCA
#=========================================================================

# -----------------------
# Step 0: Prepare TF-IDF matrix
# -----------------------
tf_idf_mat <- as.matrix(tf_idf_matrix_top10 %>% select(-doc_id))

# Optional: cluster assignments (for coloring)
set.seed(123)
tf_idf_matrix_top10$cluster <- sample(1:3, nrow(tf_idf_matrix_top10), replace = TRUE)

# -----------------------
# Plot 1: Before PCA (first two TF-IDF features)
# -----------------------
before_pca_df <- as.data.frame(tf_idf_mat[, 1:2])
colnames(before_pca_df) <- c("Feature1", "Feature2")
before_pca_df$cluster <- tf_idf_matrix_top10$cluster

ggplot(before_pca_df, aes(x = Feature1, y = Feature2, color = factor(cluster))) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Cluster Visualization Before PCA",
       x = "TF-IDF Feature 1", y = "TF-IDF Feature 2", color = "Cluster") +
  theme_minimal()

# -----------------------
# Step 1: Standardize the data for PCA
# -----------------------
tf_idf_scaled <- scale(tf_idf_mat)

# Step 2: Covariance matrix
cov_matrix <- cov(tf_idf_scaled)

# Step 3: Eigenvectors and eigenvalues
eigen_result <- eigen(cov_matrix)
eigenvectors <- eigen_result$vectors

# Step 4: Select top 2 PCs
top_eigenvectors <- eigenvectors[, 1:2]

# Step 5: Transform onto new plane
pca_manual_2d <- tf_idf_scaled %*% top_eigenvectors
pca_manual_2d <- as.data.frame(pca_manual_2d)
colnames(pca_manual_2d) <- c("PC1", "PC2")
pca_manual_2d$cluster <- tf_idf_matrix_top10$cluster

# -----------------------
# Plot 2: After PCA
# -----------------------
ggplot(pca_manual_2d, aes(x = PC1, y = PC2, color = factor(cluster))) +
  geom_point(size = 2, alpha = 0.8) +
  labs(title = "Cluster Visualization After PCA (Manual)",
       x = "PC1", y = "PC2", color = "Cluster") +
  theme_minimal()
#===============X===============X===========================X=============


