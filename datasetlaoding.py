'''
     Text Classification using Graph Convolutional Network
     @author: Shivanjali Vijaykumar Ranashing
'''
import logging
import os
from math import log
from pathlib import Path
from calculate import word_word_edges, save_as_pickle
from resources.constants import output_folder, input_folder, \
    dataset_details, output_column_filename, output_column_noOfWords, output_column_content, summary_column_noOfFiles, \
    summary_column_noOfUniqueWords, summary_column_uniqueWords, log_save_graph, log_pkl_saved, log_add_doc_node, \
    log_building_graph, text_graph_pkl_file_name, word_edge_graph_pkl_file_name, graph_document_edges, graph_no_nodes, \
    graph_word_edges, graph_no_edges, graph_document_nodes, graph_word_nodes, graph_no_document_nodes, \
    graph_no_word_nodes, graph_no_document_edges, graph_no_word_edges, graph_details, tf_idf_histogram, pmi_histogram, \
    tf_idf_histogram_title, pmi_histogram_title, summary_column_avgWordCount, output_column_noOfTrainingWords, \
    output_column_training_content, summary_column_avgWordCountForTraining
from utils import drawHistogram
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import numpy as np
import pandas as pd
from collections import OrderedDict
import re
import math
from nltk.corpus import stopwords
import csv
from itertools import combinations
import time

nltk.download('wordnet')
nltk.download('omw-1.4')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Dataset:
    class Graph:
        """
        Graph Information and its respective attributes
        """

        def __init__(self):
            self.doc_nodes = []
            self.no_doc_nodes = 0
            self.word_nodes = []
            self.no_word_nodes = 0
            self.doc_to_word_edges = []
            self.no_doc_to_word_edges = 0
            self.word_to_word_edges = []
            self.no_word_to_word_edges = 0
            self.total_nodes = 0
            self.total_edges = 0

        def setDocNodes(self, doc_nodes):
            """
            Set Document nodes
            :param doc_nodes:
            :return: None
            """
            self.doc_nodes = doc_nodes

        def getDocNodes(self):
            """
            Return document nodes
            :return: doc_nodes
            """
            return self.doc_nodes

        def setNoDocNodes(self, noOfDocNodes):
            """
            Set No of document nodes
            :param noOfDocNodes:
            :return: None
            """
            self.no_doc_nodes = noOfDocNodes

        def getNoDocNodes(self):
            """
            Get No of document nodes
            :return: no_doc_nodes
            """
            return self.no_doc_nodes

        def setWordNodes(self, word_nodes):
            """
            Set Word Nodes
            :param word_nodes:
            :return: None
            """
            self.word_nodes = word_nodes

        def getWordNodes(self):
            """
            Get Word Nodes
            :return: word_nodes
            """
            return self.word_nodes

        def setNoWordNodes(self, no_word_nodes):
            """
            Set No of word nodes
            :param no_word_nodes:
            :return: None
            """
            self.no_word_nodes = no_word_nodes

        def getNoWordNodes(self):
            """
            Get No of word Nodes
            :return: no_word_nodes
            """
            return self.no_word_nodes

        def setTotalNodes(self, totalNodes):
            """
            Set Total Number of Nodes
            :param totalNodes:
            :return: None
            """
            self.total_nodes = totalNodes

        def getTotalNodes(self):
            """
            Get Total number of nodes
            :return: total_nodes
            """
            return self.total_nodes

        def setDocWordEdges(self, doc_to_word_edges):
            """
            Set Document to word edges
            :param doc_to_word_edges:
            :return: None
            """
            self.doc_to_word_edges = doc_to_word_edges

        def getDocWordEdges(self):
            """
            Get Dcument to Word edges
            :return: doc_to_word_edges
            """
            return self.doc_to_word_edges

        def setNoDocWordEdges(self, no_doc_to_word_edges):
            """
            Set Number of Document to word edges
            :param no_doc_to_word_edges:
            :return: None
            """
            self.no_doc_to_word_edges = no_doc_to_word_edges

        def getNoDocWordEdges(self):
            """
            Get Number of Document to Word Edges
            :return: no_doc_to_word_edges
            """
            return self.no_doc_to_word_edges

        def setWordWordEdges(self, word_to_word_edges):
            """
            Set Word to Word Edges
            :param word_to_word_edges:
            :return: None
            """
            self.word_to_word_edges = word_to_word_edges

        def getWordWordEdges(self):
            """
            Get Word to word edges
            :return: word_to_word_edges
            """
            return self.word_to_word_edges

        def setNoWordWordEdges(self, no_word_to_word_edges):
            """
            Set Number of Word to word edges
            :param no_word_to_word_edges:
            :return: None
            """
            self.no_word_to_word_edges = no_word_to_word_edges

        def getNoWordWordEdges(self):
            """
            Get Number of Word to word edges
            :return: no_word_to_word_edges
            """
            return self.no_word_to_word_edges

        def setTotalEdges(self, total_edges):
            """
            Set Total number of edges
            :param total_edges:
            :return: None
            """
            self.total_edges = total_edges

        def getTotalEdges(self):
            """
            Get total number of edges
            :return: total_edges
            """
            return self.total_edges

    class Document:
        """
         Class represents Document.

         Attributes:
         -----------
         docName: String
                Document Name

         Methods:
         ---------
         setDocName(self, docName)
            Set document name

         getDocName(self)
            Get document name

        """

        def __init__(self):
            self.docName = ''
            self.words = []
            self.noOfWords = 0
            self.wordsForTraining = []
            self.noOfTrainingWords = 0

        def setDocName(self, docName):
            """
            Set Document Name
            :param docName:
            :return: None
            """
            self.docName = docName

        def getDocName(self):
            """
            Get Document Name
            :return: docName
            """
            return self.docName

        def setWords(self, words):
            """
            Set all words present in given document
            :param words:
            :return: None
            """
            self.words = words

        def getWords(self):
            """
            Get all words present in the given document
            :return: words
            """
            return self.words

        def setNoOfWords(self, noOfWords):
            """
            Set number of words in given document
            :param noOfWords:
            :return: noOfWords
            """
            self.noOfWords = noOfWords

        def getNoOfWords(self):
            """
            Get number of words present in given document
            :return:
            """
            return self.noOfWords

        def setWordsForTraining(self, wordsForTraining):
            """
            Set all words present in given document
            :param words:
            :return: None
            """
            self.wordsForTraining = wordsForTraining

        def getWordsForTraining(self):
            """
            Get all words present in the given document
            :return: words
            """
            return self.wordsForTraining


        def setNoOfTrainingWords(self, noOfTrainingWords):
            """
            Set number of words used for training
            :param noOfTrainingWords:
            :return: None
            """
            self.noOfTrainingWords = noOfTrainingWords

        def getNoOfTrainingWords(self):
            """
            Get number of training words
            :return: noOfTrainingWords
            """
            return self.noOfTrainingWords

    def __init__(self):
        """
         Class represents Dataset and its attributes

         Attributes:
         ------------
         documents: Array
                List of Document class i.e list of documents.
         cleanContent: String
                File content without special characters, numbers, stop words, removing strings less that three characters
                with lemmatization.
         all_content_array: Array
                String of array with all file content.
         noOfDocs: Number
                Number of documents.
         index_doc: List
                Document with respective its index.
         fileName: Array
                Array of file names
         tfidf: DataFrame
                Matrix of word and its respective document with its TF-IDF(Term Frequency–Inverse Document Frequency)
                calculated value.
         pmiCnt: DataFrame
                Matrix of co-occurred words with its respective calculated PMI(Point-wise Mutual Information) value.
         featureName: Array
                Array of features/classes names
        """
        self.documents = []
        self.cleanContent = ''
        self.totalUniqueWords = 0
        self.uniqueWords = []
        self.all_content_line = ''
        self.contentDict = pd.DataFrame()
        self.all_content_array = []
        self.noOfDocs = 0
        self.index_doc = {}
        self.fileNames = []
        self.tfidf = pd.DataFrame()
        self.pmiCnt = pd.DataFrame()
        self.featureNames = []
        self.graph = Dataset.Graph()

    def setnoOfDocs(self, noOfDocs):
        """
        Set Number of Documents in the Dataset
        :param noOfDocs: Number of Documents
        :return: None
        """
        self.noOfDocs = noOfDocs

    def getnoOfDocs(self):
        """
        Get Number of documents in the Dataset
        :return: noOfDocs
        """
        return self.noOfDocs

    def setCleanContent(self, cleanContent):
        """
        Set file content without special characaters, numbers, stop words, removing strings less than three characters
        with lemmatization
        :param cleanContent:
        :return: None
        """
        self.cleanContent = cleanContent

    def getCleanContent(self):
        """
        Get clean file content
        :return: cleanContent
        """
        return self.cleanContent

    def setUniqueWords(self, uniqueWords):
        """
        Set unique words from dataset
        :param uniqueWords:
        :return: None
        """
        self.uniqueWords = uniqueWords

    def getUniqueWords(self):
        """
        Get unique words from dataset
        :return: uniqueWords
        """
        return self.uniqueWords

    def setNoUniqueWords(self, uniqueWords):
        """
        Set Unique words
        :param uniqueWords:
        :return: None
        """
        self.totalUniqueWords = uniqueWords

    def getNoUniqueWords(self):
        """
        Get total number of unique words from dataset
        :return: totalWords
        """
        return self.totalUniqueWords

    def setAllContentLine(self, all_content_line):
        """
        Set all files content in one string form
        :param all_content_line:
        :return: None
        """
        self.all_content_line = all_content_line

    def getAllContentLine(self):
        """
        Get all files content in one string form
        :return: all_content_line
        """
        return self.all_content_line

    def setAllContentArray(self, all_content_array):
        """
        Set all files content in array form
        :param all_content_array:
        :return: None
        """
        self.all_content_array = all_content_array

    def getAllContentArray(self):
        """
        Get all files content in array form
        :return: all_content_array
        """
        return self.all_content_array

    def setDictContent(self, contentDict):
        """
        Set all files content in dict form
        :param contentDict:
        :return: None
        """
        self.contentDict = contentDict

    def getDictContent(self):
        return self.contentDict


    def setIndexDoc(self, index_doc):
        """
        Set document to index list
        :param index_doc:
        :return: None
        """
        self.index_doc = index_doc

    def getIndexDoc(self):
        """
        Get document to index list
        :return: index_doc
        """
        return self.index_doc

    def setDocuments(self, documents):
        """
        Set array of Documents
        :param documents:
        :return: None
        """
        self.documents = documents

    def getDocuments(self):
        """
        Get array of Documents
        :return: documents
        """
        return self.documents

    def setFileNames(self, fileNames):
        """
        Set array of file names
        :param fileNames:
        :return: None
        """
        self.fileNames = fileNames

    def getFileNames(self):
        """
        Get array of file names
        :return: fileNames
        """
        return self.fileNames

    def setTfidf(self, tfidf):
        """
        Set TF-IDF of all words in all files
        :param tfidf:
        :return: None
        """
        self.tfidf = tfidf

    def getTfidf(self):
        """
        Get TF-IDF of all words in all files
        :return:
        """
        return self.tfidf

    def setPmiCnt(self, pmiCnt):
        """
        Set PMI values of all co-occurred words
        :param pmiCnt:
        :return: None
        """
        self.pmiCnt = pmiCnt

    def getPmiCnt(self):
        """
        Get PMI values of all co-occurred words
        :return:
        """
        return self.pmiCnt

    def setfeatureNames(self, featureNames):
        """
        Set feature/class names
        :param featureNames:
        :return: None
        """
        self.featureNames = featureNames

    def getFeatureNames(self):
        """
        Get feature/class names
        :return: featureNames
        """
        return self.featureNames

    def setGraph(self, graph):
        """
        Set graph belongs to respective dataset
        :param graph:
        :return: None
        """
        self.graph = graph

    def getGraph(self):
        """
        Get Graph object of respective dataset
        :return: graph
        """
        return self.graph

    def readFilesDocCleaning(self, features):
        """
        Read input files and clean its content and preserved file content as per requirement.
        :param features: List of feature/class names
        :return: None
        """
        docs = []
        # Reading Dataset files
        source_dir = Path(input_folder)
        files = source_dir.iterdir()
        no_of_docs = 0

        # Set of stop words
        en_stops = set(stopwords.words('english'))

        all_content_array = []
        all_content_line = ""
        index_doc = {}
        file_Id = 0
        doc_nodes = []
        lemmatizer = WordNetLemmatizer()
        df_data = pd.DataFrame(columns=["content", "category"])
        start_time = time.time()
        for file in files:
            with open(file, encoding="utf-8") as fp:
                no_of_docs = no_of_docs + 1
                # List file names as document nodes
                doc_nodes.append(str(file.name))

                # Cleaned file name
                cleanedFileName = [i for i in features if i in str(file.name).lower().replace('_', ' ')]
                index_doc[file_Id] = cleanedFileName[0]
                file_Id = file_Id + 1
                # Reading file content
                current_content = fp.read()

                # Removing spaces, special characters from tokens
                content_no_spchar = re.sub(r"[^a-zA-Z]", " ", current_content).split()

                # Set Document class object
                doc = Dataset.Document()
                doc.setDocName(file.name)
                doc.setWords(content_no_spchar)
                doc.setNoOfWords(len(content_no_spchar))
                docs.append(doc)  # Keep track of document list

                list_without_stop_word = ""
                for word in content_no_spchar:
                    # Removing stop words, space and string less than three characters
                    word = lemmatizer.lemmatize(word.strip().lower())
                    if word not in en_stops and word != "" and word != " " and len(word) > 3:
                        list_without_stop_word = list_without_stop_word + " " + word
                all_content_array.append(list_without_stop_word)
                all_content_line = all_content_line + list_without_stop_word
                dum = pd.DataFrame(columns=["content", "category"])
                dum = dum.append({'content': list_without_stop_word, 'category': cleanedFileName[0]}, ignore_index=True)
                df_data = pd.concat([df_data, dum], ignore_index=True)

                doc.setWords(list_without_stop_word)
                doc.setNoOfWords(len(list_without_stop_word))
                docs.append(doc)  # Keep track of document list

        print("---Reading time %s seconds ---" % (time.time() - start_time))
        # Set all resultant values to dataset class object
        self.setCleanContent(content_no_spchar)
        self.setAllContentLine(all_content_line)
        self.setAllContentArray(all_content_array)
        self.setIndexDoc(index_doc)
        self.setDocuments(docs)
        self.setFileNames(doc_nodes)
        self.setnoOfDocs(len(docs))
        self.setDictContent(df_data)
        return df_data

    def getDatasetDetails(self):
        """
        Get Dataset Details.
        :return: None
        """
        documents = self.getDocuments()
        table = []
        row = [output_column_filename, output_column_noOfWords, output_column_noOfTrainingWords, output_column_content, output_column_training_content]
        table.append(row)
        avg_word_count = 0
        avg_word_training_count = 0
        for document in documents:
            row = []
            row.append(document.getDocName())
            row.append(document.getNoOfWords())
            row.append(document.getNoOfTrainingWords())
            row.append(document.getWords())
            row.append(document.getWordsForTraining())
            table.append(row)
            avg_word_count += document.getNoOfWords()
            avg_word_training_count += document.getNoOfTrainingWords()

        summary = []
        row = []
        row.append(summary_column_noOfFiles)
        row.append(self.getnoOfDocs())
        summary.append(row)
        row = []
        row.append(summary_column_noOfUniqueWords)
        row.append(self.getNoUniqueWords())
        summary.append(row)
        row = []
        row.append(summary_column_uniqueWords)
        row.append(self.getUniqueWords())
        summary.append(row)
        row = []
        row.append(summary_column_avgWordCount)
        row.append(avg_word_count/len(documents))
        summary.append(row)
        row = []
        row.append(summary_column_avgWordCountForTraining)
        row.append(avg_word_training_count/len(documents))
        summary.append(row)

        with open(dataset_details, 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(r) for r in table]
            [writer.writerow(r) for r in summary]

    def getGraphDetails(self):
        """
        Write Graph details (No of nodes, No of edges etc.) in csv file
        :return: None
        """
        table = []
        row = [graph_no_document_nodes, graph_no_word_nodes, graph_no_nodes,
               graph_no_document_edges, graph_no_word_edges, graph_no_edges]
        table.append(row)
        graph = self.getGraph()
        row = [graph.getNoDocNodes(), graph.getNoWordNodes(), graph.getTotalNodes(),
               graph.getNoDocWordEdges(), graph.getNoWordWordEdges(), graph.getTotalEdges()]
        table.append(row)
        row = [graph_document_nodes]
        table.append(row)
        row = []
        for docNode in graph.getDocNodes():
            row.append(docNode)
        table.append(row)
        row = [graph_word_nodes]
        table.append(row)
        row = []
        for wordNode in graph.getWordNodes():
            row.append(wordNode)
        table.append(row)
        row = [graph_document_edges]
        table.append(row)
        row = []
        for docEdge in graph.getDocWordEdges():
            row.append(docEdge)
        table.append(row)
        row = [graph_word_edges]
        table.append(row)
        row = []
        for wordEdge in graph.getWordWordEdges():
            row.append(wordEdge)
        table.append(row)

        with open(graph_details, 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(r) for r in table]

    def FrequencyCalculation(self, sliding_window_size=10):
        """
        Calculate TF-IDF of all file content, Word frequency in all files, PMI of co-occurred words
        :return: None
        """
        # Dataset Setting Values
        content_no_spchar = self.getDictContent()["content"]
        #content_no_spchar = self.getAllContentArray()
        all_content_line = self.getAllContentLine()

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(content_no_spchar)
        vocab = vectorizer.get_feature_names()
        self.setfeatureNames(vocab)
        tfidf = X.todense()
        df_tfidf = pd.DataFrame(X.toarray(), columns=np.array(vocab), index=np.array(self.getFileNames()))
        self.setTfidf(df_tfidf)

        self.setUniqueWords(vocab)
        self.setNoUniqueWords(vocab.__len__())

        # Word and its respective index
        word2index = OrderedDict((word, index) for index, word in enumerate(vocab))
        wordDict = OrderedDict((name, 0) for name in vocab)

        # Occurrance
        occurrences = np.zeros((len(vocab), len(vocab)), dtype=np.int32)

        # Find the co-occurrences:
        no_windows = 0

        all_content_array = self.getAllContentArray()
        windows = []
        for l in all_content_array:
            tokens = l.split()
            for i in range(len(tokens) - sliding_window_size):
                no_windows += 1

                d = tokens[i:(i + sliding_window_size)]
                windows.append(d)

                for w in d:
                    wordDict[w] += 1

                for w1, w2 in combinations(d, 2):
                    i1 = word2index[w1]
                    i2 = word2index[w2]

                    occurrences[i1][i2] += 1
                    occurrences[i2][i1] += 1

        # In how many windows word appear
        word_window_fre = {}
        for window in windows:
            appeared = []
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_fre:
                    word_window_fre[window[i]] += 1
                else:
                    word_window_fre[window[i]] = 1
                appeared.append(window[i])

        # In how many windows word-pair appear
        word_pair_window_fre = {}
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    str1 = window[i]
                    str2 = window[j]

                    if str1 == str2:
                        continue

                    word_pair_str1 = str1 + ',' + str2
                    if word_pair_str1 in word_pair_window_fre:
                        word_pair_window_fre[word_pair_str1] += 1
                    else:
                        word_pair_window_fre[word_pair_str1] = 1

                    word_pair_str2 = str2 + ',' + str1
                    if word_pair_str2 in word_pair_window_fre:
                        word_pair_window_fre[word_pair_str2] += 1
                    else:
                        word_pair_window_fre[word_pair_str2] = 1

        # PMI List
        pmi_list = np.zeros((len(vocab), len(vocab)))
        num_window = len(windows)
        for key in word_pair_window_fre:
            temp = key.split(',')
            word1 = temp[0]
            word2 = temp[1]
            pair_count = word_pair_window_fre[key]
            word1_count = word_window_fre[word1]
            word2_count = word_window_fre[word2]
            pmi = log((1.0 * pair_count / num_window) /
                      (1.0 * word1_count * word2_count / (num_window * num_window)))
            if pmi <= 0:
                continue
            else:
                i1 = word2index[word1]
                i2 = word2index[word2]
                pmi_list[i1][i2] = pmi
                pmi_list[i2][i1] = pmi

        pmi_dataframe = pd.DataFrame(pmi_list, index=vocab, columns=vocab)

        # PMI Calculation
        p_ij = pd.DataFrame(occurrences, index=vocab, columns=vocab) / no_windows
        p_i = pd.Series(wordDict, index=wordDict.keys()) / no_windows

        for col in p_ij.columns:
            p_ij[col] = p_ij[col] / p_i[col]
        for row in p_ij.index:
            p_ij.loc[row, :] = p_ij.loc[row, :] / p_i[row]
        p_ij = p_ij + 1E-9
        for col in p_ij.columns:
            p_ij[col] = p_ij[col].apply(lambda x: math.log(x))
        self.setPmiCnt(pmi_dataframe)

    def createGraph(self):
        """
        Create graph from saved values
        Document are used as Document Nodes
        Words are used as Word Nodes
        TF-IDF of all unique words used for Word to Document edge
        PMI of co-occurred words used for Word to Word edge
        Save Graph in pickle file
        :return: None
        """
        graph = self.getGraph()
        logger.info(log_building_graph % (
            len(self.getTfidf().index), len(self.getFeatureNames())))
        G = nx.Graph()
        logger.info(log_add_doc_node)
        G.add_nodes_from(self.getTfidf().index)  # Document Nodes
        graph.setDocNodes(self.getTfidf().index)
        graph.setNoDocNodes(len(self.getTfidf().index))
        logger.info(log_add_doc_node)
        G.add_nodes_from(self.getFeatureNames())  # Word Nodes
        graph.setWordNodes(self.getFeatureNames())
        graph.setNoWordNodes(len(self.getFeatureNames()))
        graph.setTotalNodes(graph.getNoDocNodes() + graph.getNoWordNodes())

        # Document-to-Word edges
        doc_word_edges = [(doc, word, {"weight": self.getTfidf().loc[doc, word]}) for doc in self.getTfidf().index
                          for word in self.getTfidf().columns if self.getTfidf().loc[doc, word] != 0]
        tf_idf_weights = [self.getTfidf().loc[doc, word] for doc in self.getTfidf().index
                          for word in self.getTfidf().columns if self.getTfidf().loc[doc, word] != 0]
        G.add_edges_from(doc_word_edges, color='black', weight=1)
        graph.setDocWordEdges(doc_word_edges)
        graph.setNoDocWordEdges(len(doc_word_edges))
        drawHistogram(tf_idf_weights, tf_idf_histogram, tf_idf_histogram_title)

        # Word-to-Word Edges
        words_edges = word_word_edges(self.getPmiCnt())
        pmi_weights = [word[2]['weight'] for word in words_edges]
        G.add_edges_from(words_edges, color='r', weight=2)
        graph.setWordWordEdges(words_edges)
        graph.setNoWordWordEdges(len(words_edges))
        drawHistogram(pmi_weights, pmi_histogram, pmi_histogram_title)

        graph.setTotalEdges(len(graph.getDocWordEdges()) + len(graph.getWordWordEdges()))
        if not os.listdir(output_folder).__contains__(text_graph_pkl_file_name):
            print(log_save_graph)
            save_as_pickle(word_edge_graph_pkl_file_name, words_edges)
            save_as_pickle(text_graph_pkl_file_name, G)
            logger.info(log_save_graph)
        else:
            print(log_pkl_saved)
            logger.info(log_pkl_saved)

        '''
           Coloring nodes.
           Document Nodes: Green Color
           Word Nodes: Blue Color
        '''
        for n in G.nodes():
            G.nodes[n]['color'] = 'g' if n in self.getTfidf().index else 'b'

        '''
           Fetching edge color attribute.
           Document-to-Word: Black colored edge 
           Word-to-Word: Red colored edge            
        '''
        '''colors = nx.get_edge_attributes(G, 'color').values()
        pos = nx.spring_layout(G)
        node_colors = [node[1]['color'] for node in G.nodes(data=True)]
        plt.figure(figsize=(100, 100))
        plt.title(text_graph_name)
        nx.draw_networkx(G, pos,
                         edge_color=colors,
                         with_labels=True,
                         node_color=node_colors)
        plt.savefig(text_graph_name, dpi=100)
        plt.show()'''

    def labelSetting(self):
        """
        Label words with its respective class (i.e in which document that word is occurred)
        :return: None

        """
        word_doc = {}
        for column in self.getTfidf():
            column_values = self.getTfidf()[column].values
            non_zero_index = np.array((column_values).nonzero())
            if len(non_zero_index[0]) == 1:
                word_doc[column] = self.getIndexDoc()[non_zero_index[0][0]]
            else:
                '''
                If same unique word occurred in more than one document then 
                consider highest TF-IDF value and label that word with that respective class.                
                '''
                non_zero_index_values = pd.Series(column_values).iloc[non_zero_index[0]]
                maxValue = max(non_zero_index_values)
                maxIndex = list(column_values).index(maxValue)
                word_doc[column] = self.getIndexDoc()[maxIndex]
        return word_doc
