'''
     Text Classification using Graph Convolutional Network
     @author: Shivanjali Vijaykumar Ranashing
'''

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from resources.constants import untrained_nodes, trained_nodes, data_index_distribution, \
    label_not_selected_distribution, text_graph_name
import seaborn as sns
import networkx as nx

def plotGraph(x, y, x_label, y_label, fileName, graph_title):
    """
    Plot graph for respective data
    :param x: X-axis values
    :param y: Y-axis values
    :param x_label: X-axis labels
    :param y_label: Y-axis labels
    :param graph_title: Name of graph
    :return: None
    """
    plt.figure(figsize=(15, 4))
    plt.title(graph_title, fontdict={'fontweight': 'bold', 'fontsize': 18})
    plt.plot(x, y, label=graph_title)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(fileName, dpi=100)


def plotCombined(x, y_trained, y_untrained, x_label, y_label, fileName, graph_title):
    """
    Plot Trained and Un-trained values
    :param x: x-axis values
    :param y_trained: y-axis trained values
    :param y_untrained: y-axis un-trained values
    :param x_label: Label for x-axis
    :param y_label: Label for y-axis
    :param fileName: Image file name to save plotted graph
    :param graph_title: Title of plotted graph
    :return: None
    """
    plt.figure(figsize=(15, 4))
    plt.title(graph_title, fontdict={'fontweight': 'bold', 'fontsize': 18})
    plt.plot(x, y_trained, c="red", marker="v", label=trained_nodes)
    plt.plot(x, y_untrained, c="blue", marker="o", label=untrained_nodes)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(graph_title)
    plt.savefig(fileName, dpi=100)


def labelDistribution(labels_not_selected, labels_selected, categories):
    """
    Plot label distribution of selected and not-selected data points.
    :param labels_not_selected: Labels not selected for training (Test set)
    :param labels_selected: Labels selected for the training
    :return: None
    """
    fig = plt.figure(figsize=(15, 17))
    ax = fig.add_subplot(111)
    ax.hist([(e) for e in labels_not_selected] + [(e) for e in labels_selected], bins=66)
    ax.set_title("Class label distribution for data set", fontsize=20)
    ax.set_xlabel(categories, fontsize=17)
    ax.set_ylabel("Counts", fontsize=17)
    [x.set_fontsize(15) for x in ax.get_xticklabels()]
    [x.set_fontsize(15) for x in ax.get_yticklabels()]
    plt.savefig(data_index_distribution)

    fig1 = plt.figure(figsize=(15, 17))
    ax = fig1.add_subplot(111)
    ax.hist([(e) for e in labels_not_selected], bins=66)
    ax.set_title("Class label distribution for test set", fontsize=20)
    ax.set_xlabel(categories, fontsize=17)
    ax.set_ylabel("Counts", fontsize=17)
    [x.set_fontsize(15) for x in ax.get_xticklabels()]
    [x.set_fontsize(15) for x in ax.get_yticklabels()]
    plt.savefig(label_not_selected_distribution)


def plotConfusionMatrix(confusion_matrix, imageName):
    """
    Plot confusion matrix.
    :param confusion_matrix: confusion matrix to plot
    :return: None
    """
    plt.figure(figsize=(15, 4))
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='.10g')
    ax.set_title("Confusion Matrix", fontsize=20)
    ax.set_xlabel("Actual class", fontsize=17)
    ax.set_ylabel("Predicted", fontsize=17)
    plt.savefig(imageName)


def plotTSNE(selectData, selectedLabels, notSelectedData, notSelectedLabels,
             training_title, testing_title, training_file_name, testing_file_name):
    """
        Plot output of neural network
        :param selectData: Selected training data
        :param selectedLabels: Labels of respected selected training dataset
        :param notSelectedData: Not selected training data i.e. test dataset
        :param notSelectedLabels: Labels of respected non-selected dataset for testing
        :param training_title: Title of graph for training dataset
        :param testing_title: Title of graph for testing dataset
        :param training_file_name: Image file name of plotted scatter graph of training dataset
        :param testing_file_name: Image file name of plotted scatter graph of testing dataset
        :return:
        """
    legends = ['Accounting', 'Biology', 'Geography', 'Physics', 'Computer Science', 'History', 'Math']
    z = TSNE(n_components=2).fit_transform(selectData.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    scatter = plt.scatter(z[:, 0], z[:, 1], s=70, c=selectedLabels, cmap="Set2")
    plt.title(training_title)
    plt.tight_layout()
    plt.legend(handles=scatter.legend_elements()[0], labels=legends)
    plt.savefig(testing_title)

    z = TSNE(n_components=2).fit_transform(notSelectedData.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    scatter = plt.scatter(z[:, 0], z[:, 1], s=70, c=notSelectedLabels, cmap="Set2")
    plt.title(training_file_name)
    plt.tight_layout()
    plt.legend(handles=scatter.legend_elements()[0], labels=legends)
    plt.savefig(testing_file_name)


def drawGraph(G):
    pos = nx.spring_layout(G)
    node_colors = [node[1]['color'] for node in G.nodes(data=True)]
    plt.figure(figsize=(100, 100))
    plt.title(text_graph_name)
    colors = nx.get_edge_attributes(G, 'color').values()
    nx.draw_networkx(G, pos,
                     edge_color=colors,
                     with_labels=True,
                     node_color=node_colors)
    plt.savefig(text_graph_name, dpi=100)

def plotBarChart(features, y, title, filename, categories):
    """
    Plot bar chart
    :param x: Class names
    :param y: Respective values of particular class
    :param features: Labels of x-axis
    :param title: Title of the graph
    :param filename: Filename of plotted bar chart
    :return:
    """
    plt.figure(figsize=(15, 4))
    plt.bar(features, y)
    plt.title(title)
    plt.xlabel(categories)
    plt.savefig(filename)


def plotHistogram(array, fileName, title):
    """
    Draw Histogram for given input
    :param array: Given Input for the histogram
    :param bins: Bins for the given input
    :param fileName: Name of file to save
    :return: None
    """
    plt.figure(figsize=(15, 4))
    plt.hist(array)
    plt.title(title, fontdict={'fontweight': 'bold', 'fontsize': 18})
    plt.savefig(fileName, dpi=100)
