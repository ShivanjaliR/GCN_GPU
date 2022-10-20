import os
import pickle

from resources.constants import resource_path, accuracy_file, output_folder
import math
from operator import truediv
from tqdm import tqdm
from itertools import combinations
import numpy as np

def save_as_pickle(filename, data):
    """
    Save Graph, Dataset, edges in pickle file
    :param filename: File name where you want to save graph/dataset/edges
    :param data: Data of graph/dataset/edges
    :return: None
    """
    completeName = os.path.join(output_folder, filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def read_pickel(filename):
    #filePath = os.path.join(output_folder, filename)
    #filePath = '/home/ranashsv/textclassification/GCN_GPU/input_560/'+filename
    #fileContent = pd.read_pickle(filePath)
    fileContent = pickle.load(open(filename, "rb"))

    return fileContent

def calculateMetrics(confusion_matrix):
    """
    Calculate precision, recall and f1 score of respective confusion matrix
    :param confusion_matrix: confusion matrix
    :return: precision, recall and f1 score
    """
    tp = np.diag(confusion_matrix)
    precision = list(map(truediv, tp, np.sum(confusion_matrix, axis=0)))
    recall = list(map(truediv, tp, np.sum(confusion_matrix, axis=1)))
    F1_measure = []
    for i in range(len(precision)):
        F1_measure.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
    return precision, recall, F1_measure

def generateLabels():
    """
    Read unique classes/labels from the file
    :return: None
    """
    features = []
    f = open(resource_path, "r")
    classes = f.read().split("\n")
    for feature in classes:
        features.append(feature)
    return features

def nCr(n, r):
    """
    Calculate combinations
    :param n: Number of items in list
    :param r: Number of items suppose to select from n
    :return: Calculated result
    """
    f = math.factorial
    return int(f(n) / (f(r) * f(n - r)))


def word_word_edges(p_ij):
    """
    Create word-to-word edges and assign weights to respective edge
    :param p_ij: Words Co-occurrenced (Bigram) frequency
    :return: None
    """
    word_word = []
    cols = list(p_ij.columns)
    cols = [str(w) for w in cols]
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if (p_ij.loc[w1, w2] > 0):
            word_word.append((w1, w2, {"weight": p_ij.loc[w1, w2]}))
    return word_word

def accuracy(output, labels, title):
    """
    Calculate accuracy between predicted result and actual result
    :param output: Predicted result from GCN classifier model
    :param labels: Actual labels
    :return: Number of correct labels/classes
    """
    _, prediction = output.max(1)
    prediction = prediction.numpy()
    actual_labels = [(label) for label in labels]
    correct = sum(actual_labels == prediction)

    with open(accuracy_file, "a") as fp:
        fp.write(title)
        for line in zip(actual_labels, prediction):
            fp.write(str(line))
            fp.write("\n")
    return correct / len(prediction), actual_labels, prediction

def encodeLabeles(word_class, index_doc):
    """
    Encode class labels in binary result
    :param word_class: word
    :param index_doc: Index of documents
    :return: One-hot encoding of classes
    """
    classes_dict = {feature: np.identity(len(index_doc))[index, :] for index, feature in
                    enumerate(index_doc.values())}
    labels_onehot = np.array(list(map(classes_dict.get, word_class)),
                             dtype=np.int32)
    return labels_onehot

def writeTextFile(fileName, title, lines):
    """
    Write text file
    :param fileName: Text file name
    :param title: Heading/Title of section
    :param lines: Content to write
    :return: None
    """
    with open(fileName, "a") as fp:
        fp.write(title)
        for line in lines:
            fp.write(str(line))
            fp.write("\n")