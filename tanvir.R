library(readr)
library(tokenizers)
library(dplyr)

# Read CSV
data <- read_csv("archive/7282_1_5k.csv")

# Tokenize column 'text' (replace with your actual column name)
dataset <- data %>%
  mutate(tokens = tokenize_words(text))
