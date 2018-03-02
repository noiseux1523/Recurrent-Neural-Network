# LSTM for sequence classification in the IMDB dataset
import nltk as nltk
import numpy
import sys
import csv
import itertools
import operator
import nltk
import sys
import os
import time
import pickle
from datetime import datetime
from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
csv.field_size_limit(sys.maxsize)
top_words = 6500 # Base sur notre dataset, voir method_line_count.csv dans ..../runs

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"

filename = ['/Users/noiseux1523/PycharmProjects/RNN/data/tmp_ant_train_no.csv',
            '/Users/noiseux1523/PycharmProjects/RNN/data/tmp_ant_train_yes.csv',
            '/Users/noiseux1523/PycharmProjects/RNN/data/tmp_ant_test_no.csv',
            '/Users/noiseux1523/PycharmProjects/RNN/data/ant-test-yes.csv']


# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading CSV file...")
with open('/Users/noiseux1523/PycharmProjects/RNN/data/tmp_ant_train_no.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    sentences = ["%s" % (x) for x in sentences]
print("Parsed %d sentences." % (len(list(sentences))))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data
X_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = numpy.transpose(numpy.asarray([0] * (len(list(sentences)))))
# y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading CSV file...")
with open('/Users/noiseux1523/PycharmProjects/RNN/data/tmp_ant_train_yes.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    sentences = ["%s" % (x) for x in sentences]
print("Parsed %d sentences." % (len(list(sentences))))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data
X_train_yes = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train_yes = numpy.transpose(numpy.asarray([1] * (len(list(sentences)))))

X_Train = numpy.concatenate((X_train, X_train_yes), axis=0)
y_Train = numpy.concatenate((y_train, y_train_yes), axis=0)

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading CSV file...")
with open('/Users/noiseux1523/PycharmProjects/RNN/data/tmp_ant_test_no.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    sentences = ["%s" % (x) for x in sentences]
print("Parsed %d sentences." % (len(list(sentences))))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data
X_test = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_test = numpy.transpose(numpy.asarray([0] * (len(list(sentences)))))
# y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading CSV file...")
with open('/Users/noiseux1523/PycharmProjects/RNN/data/ant-test-yes.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    sentences = ["%s" % (x) for x in sentences]
print("Parsed %d sentences." % (len(list(sentences))))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data
X_test_yes = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_test_yes = numpy.transpose(numpy.asarray([1] * (len(list(sentences)))))

X_Test = numpy.concatenate((X_test, X_test_yes), axis=0)
y_Test = numpy.concatenate((y_test, y_test_yes), axis=0)

# X_test = X_test + X_test_yes
# y_test = y_test + y_test_yes

# truncate and pad input sequences
max_review_length = 500
X_Train = sequence.pad_sequences(X_Train, maxlen=max_review_length)
X_Test = sequence.pad_sequences(X_Test, maxlen=max_review_length)

print(X_Train)
print(y_Train)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_Train, y_Train, validation_data=(X_Test, y_Test), epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_Test, y_Test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))










