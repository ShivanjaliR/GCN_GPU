'''
     Text Classification using Graph Convolutional Network
     @author: Shivanjali Vijaykumar Ranashing
'''

from __future__ import division
from __future__ import print_function
import sys
import numpy as np
from calculate import generateLabels
from datasetlaoding import Dataset
from resources.constants import output_file, test_index_file_name, \
    selected_index_file, not_selected_file, \
    selected_label_file, not_selected_label_file, training_dataset_size, testing_dataset_size, sliding_window_size, \
    selected_label_doc_index_file, \
    not_selected_label_doc_index_file, X_input, f_function, Ahat_input, graph_input, \
    selected_index_training, selected_label_doc_index, test_idxs_file, features_file
from textGraph import TextGraph
from calculate import save_as_pickle

if __name__ == '__main__':

    sys.stdout = open(output_file, "w+")

    # Step 1: Dataset Generation and Cleaning
    dataset = Dataset()
    features = generateLabels()

    save_as_pickle(features_file, features)

    contentDict = dataset.readFilesDocCleaning(features)
    index_doc = dataset.getIndexDoc()

    # Step 2: Frequency Calculation
    dataset.FrequencyCalculation(sliding_window_size)

    # Step 1.2: Dataset Details
    dataset.getDatasetDetails()

    # Step 3: Creating Graph
    dataset.createGraph()

    # Step 4: Labeling words
    wordClasses = dataset.labelSetting()

    node_labels_values = list(index_doc.values())

    node_labels = []
    for node_label in node_labels_values:
        if node_label in features:
            node_labels.append(features.index(node_label))

    classes = wordClasses.values()
    word_labels = [features.index(cls) for cls in classes]
    all_labels = list(node_labels) + word_labels

    # Step 5: Split Training and Testing Dataset
    test_idxs = []
    test_ratio = .20
    for cls in contentDict["category"].unique():
        dum = contentDict[contentDict["category"] == cls]["category"]
        if len(dum) >= 4:
            test_idxs.extend(list(np.random.choice(dum.index, size=round(test_ratio * len(dum)), replace=False)))

    save_as_pickle(test_index_file_name, test_idxs)
    # select only certain labelled nodes for semi-supervised GCN
    selected = []
    not_selected = []
    for i in range(len(contentDict)):
        if i not in test_idxs:
            selected.append(i)
        else:
            not_selected.append(i)


    labels_selected = [l for idx, l in enumerate(contentDict["category"]) if idx in selected]
    labels_not_selected = [l for idx, l in enumerate(contentDict["category"]) if idx not in selected]
    labels_selected_doc_index = [features.index(labels_selected[idx]) for idx in range(len(labels_selected))]
    labels_not_selected_doc_index = [features.index(labels_not_selected[idx]) for idx in
                                     range(len(labels_not_selected))]

    '''Dumping training and testing dataset'''

    save_as_pickle(selected_index_file, selected)
    save_as_pickle(not_selected_file, not_selected)

    save_as_pickle(selected_label_file, labels_selected)
    save_as_pickle(not_selected_label_file, labels_not_selected)

    save_as_pickle(selected_label_doc_index_file, labels_selected_doc_index)
    save_as_pickle(not_selected_label_doc_index_file, labels_not_selected_doc_index)

    save_as_pickle(test_idxs_file, test_idxs)

    print(training_dataset_size, len(labels_selected_doc_index))
    print(testing_dataset_size, len(labels_not_selected_doc_index))

    # Step 6. Reading Graph and fetching its respective attributes
    textGraph = TextGraph()
    f, X, A_hat, graph = textGraph.loadGraph()

    save_as_pickle(X_input, X)
    save_as_pickle(f_function, f)
    save_as_pickle(Ahat_input, A_hat)
    save_as_pickle(graph_input, graph)

    save_as_pickle(selected_index_training, selected)
    save_as_pickle(selected_label_doc_index, labels_selected_doc_index)


    # Step 7. Graph Details
    dataset.getGraphDetails()