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

import os
import numpy
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
#import matplotlib.pyplot as plt
import logging

numpy.set_printoptions(threshold = numpy.nan)

# Function to visualize de confusion matrix
#   data : The confusion matrix
#   title : Title of the plot
#   cmap : Color map
#   name : Name of the dataset
#def plot_confusion_matrix(data, title = '_Confusion Matrix_', cmap = plt.cm.Blues, name = ''):
#    logging.debug("Producing Chart - Confusion Matrix - {}".format(name))
#
#    plt.imshow(data, interpolation = 'nearest', cmap = cmap)
#    plt.title(title)
#    plt.colorbar()
#    labels = numpy.array(['Negative', 'Positive'])
#    tick_marks = numpy.arange(len(labels))
#    plt.xticks(tick_marks, labels, rotation = 45)
#    plt.yticks(tick_marks, labels)
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    plt.savefig("./Plots/" + name + title + '.png', bbox_inches = 'tight')

# Function to visualize the Receiver Operating Characteristic curve
#   fpr : False Positive Rate
#   tpr : True Positive Rate
#   roc_auc : Receiver Operating Characteristic Area Under the Curve
#   name : Name of the dataset
#def plot_roc_curve(fpr, tpr, roc_auc, name = ''):
#    logging.debug("Producing Chart - ROC Curve - {}".format(name))
#
#    plt.figure()
#    plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % roc_auc)
#    plt.plot([0, 1],
#             [0, 1],
#             'k--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver Operating Characteristic')
#    plt.legend(loc = "lower right")
#    plt.savefig("./Plots/" + name + '_ROC_Curve.png', bbox_inches = 'tight')


# Fix random seed for reproducibility
numpy.random.seed(7)

# Load the dataset but only keep the top n words (vocabulary_size), zero the rest
csv.field_size_limit(sys.maxsize)
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
print("Using vocabulary size %d.\n" % vocabulary_size)

# Model parameters
embedding_vector_length = 300
LSTM_units = 150
batch_size = 64
nb_epochs = 3
dropout_rate = 0.6
max_review_length = 2000  # Base sur notre dataset, voir method_line_count.csv dans ..../runs

###########
# MODEL 1 #
###########

# Create model
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_vector_length, input_length=max_review_length))
# model.add(Bidirectional(LSTM(LSTM_units)))
model.add(Bidirectional(LSTM(LSTM_units, return_sequences=True)))
model.add(Dropout(dropout_rate))
# model.add(Bidirectional(LSTM(LSTM_units, return_sequences=True)))
# model.add(Dropout(dropout_rate))
model.add(Bidirectional(LSTM(LSTM_units, return_sequences=False)))
model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# Try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# Print information to output file
print("Embedding Vector Length;LSTM units;Batch Size;Epochs;Dropout")
print("Parameters = {0};{1};{2};{3};{4}".format(embedding_vector_length, LSTM_units, batch_size, nb_epochs, dropout_rate))
print("Accuracy;F1;Precision;Recall;LogLoss")

# 10 fold cross-validation
for it in range(0,417):
    print("Fold {}\n".format(it))

    # Create train and test files
    for file in range(0,4):

        # Create the training data (Negative Examples)
        if (file == 0):
            filename = "{}{}{}".format('/gs/scratch/cenoi/RNN/data/OracleV5C14-training/', it, '-train-lapd-no.neg')

        # Create the training data (Positive Examples)
        elif (file == 1):
            filename = "{}{}{}".format('/gs/scratch/cenoi/RNN/data/OracleV5C14-training/', it, '-train-lapd-yes.pos')

        # Create the testing data (Negative Examples)
        elif (file == 2):
            filename = "{}{}{}".format('/gs/scratch/cenoi/RNN/data/OracleV5C14-training/', it, '-test-lapd-no.neg')

        # Create the testing data (Positive Examples)
        else:
            filename = "{}{}{}".format('/gs/scratch/cenoi/RNN/data/OracleV5C14-training/', it, '-test-lapd-yes.pos')

        # Read the data, append SENTENCE_START and SENTENCE_END tokens, and parse into sentences
        print("\nReading CSV file -> %s" % filename)
        if os.stat(filename).st_size == 0:
            print("File is empty!")
        else:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                #next(reader)

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

            #print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

            # Replace all words not in our vocabulary with the unknown token
            for i, sent in enumerate(tokenized_sentences):
                tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

        # X_Train : Array of tokenized methods for training
        # y_Train : Array of labels for training
        # X_Test : Array of tokenized methods for testing
        # y_Test : Array of labels for testing

        # Create the training data (Negative Examples)
        if (file == 0):
            X_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
            y_train = numpy.transpose(numpy.asarray([0] * (len(list(sentences)))))

        # Create the training data (Positive Examples)
        elif (file == 1):
            X_train_yes = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
            y_train_yes = numpy.transpose(numpy.asarray([1] * (len(list(sentences)))))

            # Combine negative and positive examples
            X_Train = numpy.concatenate((X_train, X_train_yes), axis = 0)
            y_Train = numpy.concatenate((y_train, y_train_yes), axis = 0)

        # Create the testing data (Negative Examples)
        elif (file == 2):
            if os.stat(filename).st_size != 0:
                X_Test = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
                y_Test = numpy.transpose(numpy.asarray([0] * (len(list(sentences)))))

        # Create the testing data (Positive Examples)
        else:
            if os.stat(filename).st_size != 0:
                X_Test = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
                y_Test = numpy.transpose(numpy.asarray([1] * (len(list(sentences)))))

            # Combine negative and positive examples
            #X_Test = numpy.concatenate((X_test, X_test_yes), axis = 0)
            #y_Test = numpy.concatenate((y_test, y_test_yes), axis = 0)

    # Truncate and pad input sequences
    X_Train = sequence.pad_sequences(X_Train, maxlen = max_review_length)
    X_Test = sequence.pad_sequences(X_Test, maxlen = max_review_length)

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
    logLoss = log_loss(y_Test, pred)
    recall = recall_score(y_Test, pred)
    fpr, tpr, _ = roc_curve(y_Test, pred)
    cm = confusion_matrix(y_Test, pred)
    roc_auc = auc(fpr, tpr)
    #plot_confusion_matrix(cm, name = "dataset_name")
    #plot_roc_curve(fpr, tpr, roc_auc, name="dataset_name")

    # [ TN FN ]
    # [ FP TP ]

    print("Performance = {0};{1};{2};{3};{4}\n".format(acc_score * 100, F1_score * 100, precision * 100, recall * 100, logLoss))
    print('DONE\n')
