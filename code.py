import nltk
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from convokit import Corpus, download
import matplotlib.pyplot as plt
from collections import Counter
import ast

# Load the Cornell Movie--Dialogs Corpus
movie_corpus = Corpus(filename='/Users/adnanchowdhury/Documents/AIT526/term_project/movie-corpus')
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

# Display the genre frequencies
print(genre_freq_df)



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
