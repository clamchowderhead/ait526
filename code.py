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

# Load the Cornell Movie--Dialogs Corpus
movie_corpus = movie_corpus = Corpus(filename = download("movie-corpus"))
# movie_corpus = [utterance.text for utterance in movie_corpus.iter_utterances()]
movie_corpus.print_summary_stats()

# For each conversation in the movie corpus, get a frequency distribution of genre
df = movie_corpus.get_conversations_dataframe()

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

# Change numeric values to actual numbers

df['meta.rating'] = pd.to_numeric(df['meta.rating'])

df['meta.votes'] = pd.to_numeric(df['meta.votes'])

# Get a scatterplot between ratings and votes
grouped_df = df.groupby(['meta.movie_name'],
                        as_index = False)['meta.rating', 'meta.votes'].apply(lambda x: np.unique(x.values.ravel()).tolist())

grouped_df.reset_index(inplace = True)

plt.scatter(x = grouped_df['meta.rating'], y = grouped_df['meta.votes'], s = 1.5)
plt.show()

# # Get dataframe head
# print(convo_df.head())
# genre_freqs = movie_corpus.get_conversations_dataframe()
# print(type(genre_freqs))
# print(genre_freqs)

# # Load a hypothetical everyday language corpus
# everyday_corpus = Corpus(filename=download('wiki-corpus'))
# everyday_corpus = [utterance.text for utterance in everyday_corpus.iter_utterances()]

# print("Now processing...")

# # Define stop words
# stop_words = set(stopwords.words('english'))

# # Tokenize and clean the movie corpus
# movie_words = [word for word in nltk.word_tokenize(' '.join(movie_corpus)) if word.isalnum() and word not in stop_words]

# # Tokenize and clean the everyday language corpus
# everyday_words = [word for word in nltk.word_tokenize(' '.join(everyday_corpus)) if word.isalnum() and word not in stop_words]

# # Calculate word frequencies
# movie_word_freq = Counter(movie_words)
# everyday_word_freq = Counter(everyday_words)

# # Identify distinct words in movie scripts and sort them by the difference in frequency
# distinct_words = sorted([word for word in movie_word_freq if movie_word_freq[word] > everyday_word_freq[word]], key=lambda x: movie_word_freq[x] - everyday_word_freq[x], reverse=True)

# # Select the top 10 most distinct words
# top_distinct_words = distinct_words[:10]

# # Visualize the distinct words
# plt.figure(figsize=(10,5))
# plt.bar(top_distinct_words, [movie_word_freq[word] for word in top_distinct_words], width=0.5)
# plt.xlabel('Distinct Words')
# plt.ylabel('Frequency')
# plt.title('Top 10 Distinct Words in Movie Scripts')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# Perform sentiment analysis on each of the categories

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

# N-grams, we will go with 4-grams for a test
n_grams = ngrams(movie_words, 4)
four_grams = [' '.join(grams) for grams in n_grams]
print(four_grams[1:50])
