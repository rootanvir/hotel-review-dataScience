library(readr)
library(tokenizers)
library(dplyr)
library(purrr)
library(tidytext)

# Read CSV
data <- read_csv("D:/data science final project/archive/7282_1_5k.csv")

# 1️⃣ Tokenization dataset (all tokens, one row per review)
tokenization <- data %>%
  mutate(tokens = map(reviews.text, ~ {
    words <- tokenize_words(.x)[[1]]   # tokenize
    tolower(words)                      # convert to lowercase
  })) %>%
  select(tokens)

# 2️⃣ Tokenization without stop words dataset
tokenization_no_stop <- data %>%
  mutate(tokens = map(reviews.text, ~ {
    words <- tokenize_words(.x)[[1]]                 # tokenize
    words <- tolower(words)                           # lowercase
    words[!words %in% get_stopwords()$word]          # remove stop words
  })) %>%
  select(tokens)

# word and frequency
word_freq <- tokenization_no_stop %>%
  unnest(tokens) %>%       
  group_by(tokens) %>%     
  summarise(frequency = n(), .groups = 'drop') %>%  
  arrange(desc(frequency)) %>%    
  rename(word = tokens)


# Term Frequency per word per document
tf_data <- tokenization_no_stop %>%
  mutate(doc_id = row_number()) %>%       # Assign unique ID for each review
  unnest(tokens) %>%
  count(doc_id, tokens, name = "term_count") %>% # count term per document
  group_by(doc_id) %>%
  mutate(tf = term_count / sum(term_count)) %>%  # TF = term_count / total terms in doc
  ungroup() %>%
  rename(word = tokens)



# Number of documents
total_docs <- nrow(tokenization_no_stop)

# Document frequency (number of docs containing the word)
idf_data <- tokenization_no_stop %>%
  mutate(doc_id = row_number()) %>%
  unnest(tokens) %>%
  distinct(doc_id, tokens) %>%
  count(tokens, name = "doc_freq") %>%
  mutate(idf = log(total_docs / doc_freq)) %>%  # IDF formula
  rename(word = tokens)

