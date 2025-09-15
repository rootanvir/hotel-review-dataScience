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

# Optional: view first few rows
head(tokenization)
head(tokenization_no_stop)
