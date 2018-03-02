# coding: utf-8

# Recurrent Neural Networks Tutorial, Part 2 – Implementing a RNN with Python, Numpy and Theano
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

# Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras
# http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

# Keras Cheat Sheet: Neural Networks in Python
# https://www.datacamp.com/community/blog/keras-cheat-sheet#gs.8ZrDgrg

# Text Classification, Part 2 - sentence level Attentional RNN
# https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/

# LSTM for sequence classification in the IMDB dataset
import getopt
import os
import numpy as np
import csv
import itertools
import nltk
import sys
import random
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.metrics import roc_curve, auc, recall_score, log_loss
import matplotlib.pyplot as plt
import logging
np.set_printoptions(threshold = np.nan)


# Function to visualize de confusion matrix
#   data : The confusion matrix
#   title : Title of the plot
#   cmap : Color map
#   name : Name of the dataset
def plot_confusion_matrix(data, title = '_Confusion Matrix_', cmap = plt.cm.Blues, name = ''):
    logging.debug("Producing Chart - Confusion Matrix - {}".format(name))

    plt.imshow(data, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    labels = np.array(['Negative', 'Positive'])
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation = 45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./Plots/" + name + title + '.png', bbox_inches = 'tight')


# Function to visualize the Receiver Operating Characteristic curve
#   fpr : False Positive Rate
#   tpr : True Positive Rate
#   roc_auc : Receiver Operating Characteristic Area Under the Curve
#   name : Name of the dataset
def plot_roc_curve(fpr, tpr, roc_auc, name = ''):
    logging.debug("Producing Chart - ROC Curve - {}".format(name))

    plt.figure()
    plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1],
             [0, 1],
             'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = "lower right")
    plt.savefig("./Plots/" + name + '_ROC_Curve.png', bbox_inches = 'tight')

########################################################################################################
# PARAMETERS
########################################################################################################

# Pass parameter values
try:
    iteration=1
    embedding_dimensions=100
    LSTM_units=[250,100,50]
    batch_size=32
    nb_epochs=1
    dropout_rate=0.5
    max_input_length=0
    vocabulary_size=10000

    opts, args = getopt.getopt(sys.argv[1:], "", ['iteration=',
                                                  'embedding-dimension=',
                                                  'LSTM-units=',
                                                  'batch-size=',
                                                  'nb-epochs=',
                                                  'dropout=',
                                                  'max-input-length=',
                                                  'vocabulary-size='])

except getopt.GetoptError:
    print('RNN-complete.py \n'
          '\t--iteration <int> \n'
          '\t--embedding-dimension <int> \n'
          '\t--LSTM-units <array> \n'
          '\t--batch-size <int> \n'
          '\t--nb-epochs <int> \n'
          '\t--dropout <float> \n'
          '\t--max-input-length <int>\n'
          '\t--vocabulary-size <int>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("--iteration"):
        iteration = arg
    elif opt in ("--embedding-dimension"):
        embedding_dimensions = int(arg)
    elif opt in ("--LSTM-units"):
        LSTM_units = list(map(int, arg.split(",")))
    elif opt in ("--batch-size"):
        batch_size = int(arg)
    elif opt in ("--nb-epochs"):
        nb_epochs = int(arg)
    elif opt in ("--dropout"):
        dropout_rate = float(arg)
    elif opt in ("--max-input-length"):
        max_input_length = int(arg)
    elif opt in ("--vocabulary-size"):
        vocabulary_size = int(arg)
    else:
        print("{} is an invalid parameter".format(opt))

########################################################################################################
# GENERATE DATASETS
########################################################################################################

# Fix random seed for reproducibility
np.random.seed(7)
unknown_token = "UNKNOWN_TOKEN"
print("Fold {}\n".format(iteration))

# Create train and test files
for file in range(0, 4):

    # Create the training data (Negative Examples)
    if (file == 0):
        filename = "{}{}{}".format('/Users/noiseux1523/PycharmProjects/RNN/OracleV5C14-training/', iteration,
                                   '-train-lapd-no.neg')

    # Create the training data (Positive Examples)
    elif (file == 1):
        filename = "{}{}{}".format('/Users/noiseux1523/PycharmProjects/RNN/OracleV5C14-training/', iteration,
                                   '-train-lapd-yes.pos')

    # Create the testing data (Negative Examples)
    elif (file == 2):
        filename = "{}{}{}".format('/Users/noiseux1523/PycharmProjects/RNN/OracleV5C14-training/', iteration,
                                   '-test-lapd-no.neg')

    # Create the testing data (Positive Examples)
    else:
        filename = "{}{}{}".format('/Users/noiseux1523/PycharmProjects/RNN/OracleV5C14-training/', iteration,
                                   '-test-lapd-yes.pos')

    # Read the data, append SENTENCE_START and SENTENCE_END tokens, and parse into sentences
    print("Reading CSV file -> %s" % filename)
    with open(filename, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)

        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        sentences = ["%s" % (x) for x in sentences]

    print("Parsed %d sentences." % (len(list(sentences))))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

    print("Found %d unique words tokens.\n" % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # X_Train : Array of tokenized methods for training
    # y_Train : Array of labels for training
    # X_Test  : Array of tokenized methods for testing
    # y_Test  : Array of labels for testing

    # Create the training data (Negative Examples)
    if (file == 0):
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.transpose(np.asarray([0] * (len(list(sentences)))))

    # Create the training data (Positive Examples)
    elif (file == 1):
        X_train_yes = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train_yes = np.transpose(np.asarray([1] * (len(list(sentences)))))

        # Combine negative and positive examples
        X_Train = np.concatenate((X_train, X_train_yes), axis=0)
        y_Train = np.concatenate((y_train, y_train_yes), axis=0)

    # Create the testing data (Negative Examples)
    elif (file == 2):
        if (os.stat(filename).st_size != 0):
            X_Test = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
            y_Test = np.transpose(np.asarray([0] * (len(list(sentences)))))

    # Create the testing data (Positive Examples)
    else:
        if (os.stat(filename).st_size != 0):
            X_Test = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
            y_Test = np.transpose(np.asarray([1] * (len(list(sentences)))))

########################################################################################################
# CREATE MODEL
########################################################################################################

# Define the max length of the input if not specified by user (use the length equivalent to the 95% biggest file)
if max_input_length == 0:
    file_lengths = [len(x) for x in X_Train]
    file_lengths.sort()
    top_95 = int(len(file_lengths) * 0.95)
    max_input_length = file_lengths[top_95]
    print("Using input size %d.\n" % max_input_length)

print("Using vocabulary size %d.\n" % vocabulary_size)

# Truncate and pad input sequences
X_Train = sequence.pad_sequences(X_Train, maxlen = max_input_length)
X_Test = sequence.pad_sequences(X_Test, maxlen = max_input_length)

# Build Model
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dimensions, input_length=max_input_length)) # Embedding Layer
for i, units in enumerate(LSTM_units): # LSTM Layers + Dropout
    if i < (len(LSTM_units) - 1):
        model.add(Bidirectional(LSTM(units, return_sequences=True)))
        model.add(Dropout(dropout_rate))
    else:
        model.add(Bidirectional(LSTM(units, return_sequences=False)))
        model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid')) # Activation Function
print(model.summary())

# Try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# Print information to output file
print("Embedding Vector Length;LSTM units;Batch Size;Epochs;Dropout")
print("Parameters = {0};{1};{2};{3};{4}".format(embedding_dimensions, LSTM_units, batch_size, nb_epochs, dropout_rate))
print("Accuracy;F1;Precision;Recall;LogLoss")

# Train model
print('\nTrain...\n')
model.fit(X_Train, y_Train, batch_size = batch_size, epochs = nb_epochs, validation_data = [X_Test, y_Test])

# Final evaluation of the model
scores = model.evaluate(X_Test, y_Test, verbose = 1)
pred = model.predict_classes(X_Test, batch_size, verbose = 1)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(scores)
print(pred)

# Performance metrics
acc_score = accuracy_score(y_Test, pred)
F1_score = f1_score(y_Test, pred)
precision = precision_score(y_Test, pred)
#logLoss = log_loss(y_Test, pred)
recall = recall_score(y_Test, pred)
fpr, tpr, _ = roc_curve(y_Test, pred)
cm = confusion_matrix(y_Test, pred)
#roc_auc = auc(fpr, tpr)
#plot_confusion_matrix(cm, name = "dataset_name")
#plot_roc_curve(fpr, tpr, roc_auc, name="dataset_name")

print("TN - FP - FN - TP")
print("Test-{};{};{};{}".format(cm[0][0], cm[0][1], cm[1][0], cm[1][1]))

# [ TN FP ]
# [ FN TP ]

print("Performance = {0};{1};{2};{3}\n".format(acc_score * 100, F1_score * 100, precision * 100, recall * 100))
print('DONE\n')
