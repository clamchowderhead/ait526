# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:51:41 2023

"""
import nltk
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from convokit import Corpus, download
import matplotlib.pyplot as plt
from collections import Counter
import ast
import numpy as np
import seaborn as sns
from textblob import TextBlob
from nltk import ngrams
from wordcloud import WordCloud
import random

# Load the Cornell Movie--Dialogs Corpus
movie_corpus = movie_corpus = Corpus(filename = download("movie-corpus"))
# movie_corpus = [utterance.text for utterance in movie_corpus.iter_utterances()]
movie_corpus.print_summary_stats()

# For each conversation in the movie corpus, get a frequency distribution of genre
df = movie_corpus.get_conversations_dataframe()

# Change numeric values to actual numbers
df['meta.rating'] = pd.to_numeric(df['meta.rating'])
df['meta.votes'] = pd.to_numeric(df['meta.votes'])

# Assuming df is your DataFrame
# If meta.genre is a string representation of a list, convert it to an actual list
if df['meta.genre'].dtype == 'O':  # 'O' means object, typically strings
    df['meta.genre'] = df['meta.genre'].apply(ast.literal_eval)

# Initialize a Counter object to hold the genre frequencies
genre_freq = Counter()

# Iterate over the DataFrame rows and update the Counter with the genres
for genres in df['meta.genre']:
    genre_freq.update(genres)

# Convert the Counter to a DataFrame for better visualization
genre_freq_df = pd.DataFrame.from_dict(genre_freq, orient='index', columns=['Frequency']).reset_index()
genre_freq_df.rename(columns={'index': 'Genre'}, inplace=True)

# Display the movie frequencies with seaborn
sns.countplot(data = df, y = "meta.movie_name", order = df['meta.movie_name'].value_counts().iloc[:10].index)
plt.show()

# Change numeric values to actual numbers

df['meta.rating'] = pd.to_numeric(df['meta.rating'])

df['meta.votes'] = pd.to_numeric(df['meta.votes'])

# Get the utterances dataframe
df2 = movie_corpus.get_utterances_dataframe()

# Get the corpus that we will be using for the sentiment analysis
movie_corpus = [utterance.text for utterance in movie_corpus.iter_utterances()]

#  Stop words
stop_words = set(stopwords.words('english'))

# Tokenize and clean the movie corpus
movie_words = [word for word in nltk.word_tokenize(' '.join(movie_corpus)) if word.isalnum() and word not in stop_words]

# Create two empty lists to store the values
text_list = []
Sentiment_list = []

for text in movie_corpus:
    text_list.append(text)
    Sentiment_list.append(TextBlob(text).sentiment.polarity)
    
sent_df = pd.DataFrame(
    {
     'text' : text_list,
     'Sentiment' : Sentiment_list
     }
    )

# Filter where sentiment is 0 as we are not going to take those values into consideration
sent_df = sent_df[sent_df['Sentiment'] != 0]

# Drop duplicates for top and bottom 20
sentiment_graph = sent_df.drop_duplicates()
# Sort the dataframe
sentiment_graph.sort_values(by = ['Sentiment'], inplace = True)

# Set seed
random.seed(25)

# N-grams, we will go with 4-grams for a test
n_grams = ngrams(movie_words, 4)
four_grams = [' '.join(grams) for grams in n_grams]
print(four_grams[1:50])

# Try 2-grams for simplicity to see the results

n2_grams = ngrams(movie_words, 2)
bi_grams = [' '.join(grams) for grams in n2_grams]
print(bi_grams[1:50])

# For more complicated sentences, try 8-grams

n8_grams = ngrams(movie_words, 8)
eight_grams = [' '.join(grams) for grams in n8_grams]
print(eight_grams[1:50])

# Generate the positive sentiment wordcloud
wordcloud = WordCloud()

pos_sent_w_df = sent_df[sent_df['Sentiment'] <= 0.95]
pos_sent_w = pos_sent_w_df['text']
wordcloud.generate(' '.join(pos_sent_w))

# Display the positive wordcloud
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# Generate the negative sentiment wordcloud
wordcloud = WordCloud()

neg_sent_w_df = sent_df[sent_df['Sentiment'] >= -0.95]
neg_sent_w = neg_sent_w_df['text']
wordcloud.generate(' '.join(pos_sent_w))

# Display the positive wordcloud
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# Import the everyday corpus
everyday_corpus = Corpus(filename=download('wiki-corpus'))
everyday_corpus = [utterance.text for utterance in everyday_corpus.iter_utterances()]

# Tokenize and clean the everyday language corpus
everyday_words = [word for word in nltk.word_tokenize(' '.join(everyday_corpus)) if word.isalnum() and word not in stop_words]

# Calculate word frequencies
movie_word_freq = Counter(movie_words)
everyday_word_freq = Counter(everyday_words)

# Identify distinct words in movie scripts and sort them by the difference in frequency
distinct_words = sorted([word for word in movie_word_freq if movie_word_freq[word] > everyday_word_freq[word]], key=lambda x: movie_word_freq[x] - everyday_word_freq[x], reverse=True)

# Select the top 20 most distinct words
top_distinct_words = distinct_words[:20]

# # Visualize the distinct words
plt.figure(figsize=(10,5))
plt.bar(top_distinct_words, [movie_word_freq[word] for word in top_distinct_words], width=0.5)
plt.xlabel('Distinct Words')
plt.ylabel('Frequency')
plt.title('Top 10 Distinct Words in Movie Scripts')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create the wordcloud for the distinct words as well, but using all of them
wordcloud = WordCloud()
wordcloud.generate(' '.join(distinct_words))

# Display the difference wordcloud, this wordcloud shows where movies used a word more than everyday speech
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# Display the movie frequencies with seaborn for the top 15 movies
sns.countplot(data = df, y = "meta.movie_name", order = df['meta.movie_name'].value_counts().iloc[:15].index)
plt.show()

# Get a scatterplot between ratings and votes
grouped_df = df[['meta.movie_name', 'meta.rating', 'meta.votes']].drop_duplicates()

# Get a scatterplot between ratings and votes
sns.scatterplot(data = grouped_df, x = 'meta.rating', y = 'meta.votes')
