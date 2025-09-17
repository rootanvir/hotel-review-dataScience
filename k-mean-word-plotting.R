library(readr)
library(tokenizers)
library(dplyr)
library(purrr)
library(tidytext)
library(tidyr)
library(tibble)
library(ggplot2)
library(factoextra)

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
  left_join(word_freq, by = "word")

# =======================
# Top 10 TF-IDF table
# =======================
tf_idf_top10 <- tf_idf_full %>%
  filter(word %in% top_words)

# =======================
# TF-IDF matrix for top 10 words (docs × words)
# =======================
tf_idf_matrix_top10 <- tf_idf_top10 %>%
  select(doc_id, word, tf_idf) %>%
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%
  as_tibble()

# =======================
# Transpose so rows = words
# =======================
word_matrix <- as.data.frame(t(tf_idf_matrix_top10 %>% select(-doc_id)))
colnames(word_matrix) <- paste0("doc", 1:ncol(word_matrix))
word_matrix$word <- rownames(word_matrix)
rownames(word_matrix) <- NULL

# =======================
# K-means clustering on words
# =======================
set.seed(123)
k <- 4
kmeans_result <- kmeans(word_matrix %>% select(-word), centers = k, nstart = 25)

# =======================
# PCA for visualization
# =======================
pca_result <- prcomp(word_matrix %>% select(-word), scale. = TRUE)
pca_data <- as.data.frame(pca_result$x[, 1:2])
pca_data$cluster <- factor(kmeans_result$cluster)
pca_data$word <- word_matrix$word

# =======================
# Merge frequency for word sizes
# =======================
pca_data <- pca_data %>%
  left_join(word_freq %>% filter(word %in% top_words), by = "word")

# =======================
# Plot clusters with words as labels (size by frequency)
# =======================
ggplot(pca_data, aes(x = PC1, y = PC2, color = cluster)) +
  geom_text(aes(label = word, size = frequency), fontface = "bold") +
  scale_size(range = c(4, 10)) +
  labs(title = "K-means Clustering of Top 10 Words (TF-IDF)",
       size = "Word Frequency") +
  theme_minimal()
