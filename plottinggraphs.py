import torch
from sklearn.metrics import confusion_matrix
from calculate import read_pickel, calculateMetrics
from resources.constants import accuracy_per_epochs_file, loss_per_epochs_file, trained_loss_file, \
    trained_accuracy_file, trained_epochs_file, untrained_accuracy_file, untrained_loss_file, \
    first_layer_after_training_for_testing_dataset_file_name, first_layer_after_training_for_training_dataset_file_name, \
    first_layer_after_training_for_testing_dataset_name, first_layer_after_training_for_training_dataset_name, \
    plot_x_axis, plot_y_axis_loss, training_loss_plot_file_name, training_loss_plot_name, plot_y_axis_accuracy, \
    training_accuracy_plot_file_name, training_accuracy_plot_name, testing_loss_plot_name, testing_loss_plot_file_name, \
    combined_accuracy_plot_file_name, combined_accuracy_plot_name, X_input, test_idxs_file, \
    untrained_confusion_matrix_file_name, testing_precision_plot_name, testing_precision_plot_file_name, \
    testing_recall_plot_name, testing_recall_plot_file_name, categories_for_classification, testing_f1score_plot_name, \
    testing_f1score_plot_file_name, trained_confusion_matrix_file_name, training_precision_plot_name, \
    training_recall_plot_file_name, training_precision_plot_file_name, training_recall_plot_name, \
    training_f1score_plot_name, training_f1score_plot_file_name, features_file, num_of_epochs, \
    model_filename, selected_index_file, selected_label_doc_index, not_selected_file, \
    not_selected_label_file
from utils import plotGraph, plotCombined, plotConfusionMatrix, plotBarChart, labelDistribution, plotTSNE, drawGraph

if __name__ == '__main__':
    '''completeName = os.path.join(output_folder, text_graph_pkl_file_name)
    with open(completeName, 'rb') as pkl_file:
        G = pickle.load(pkl_file)

    drawGraph(G)'''

    model = read_pickel(model_filename)
    selected = read_pickel(selected_index_file)
    labels_selected_doc_index = read_pickel(selected_label_doc_index)
    not_selected = read_pickel(not_selected_file)
    labels_not_selected_doc_index = read_pickel(not_selected_label_file)

    features = read_pickel(features_file)

    labelDistribution(labels_not_selected_doc_index, labels_selected_doc_index, features)

    X = read_pickel(X_input)
    model.eval()
    with torch.no_grad():
        pred_labels = model(X)
    plotTSNE(pred_labels[selected], labels_selected_doc_index, pred_labels[not_selected], labels_not_selected_doc_index,
             first_layer_after_training_for_training_dataset_name,
             first_layer_after_training_for_testing_dataset_name,
             first_layer_after_training_for_training_dataset_file_name,
             first_layer_after_training_for_testing_dataset_file_name)

    loss_per_epochs = read_pickel(loss_per_epochs_file)
    accuracy_per_epochs = read_pickel(accuracy_per_epochs_file)

    plotGraph(range(num_of_epochs), loss_per_epochs, plot_x_axis, plot_y_axis_loss, training_loss_plot_file_name,
              training_loss_plot_name)
    plotGraph(range(num_of_epochs), accuracy_per_epochs, plot_x_axis, plot_y_axis_accuracy,
              training_accuracy_plot_file_name, training_accuracy_plot_name)

    trained_epochs = read_pickel(trained_epochs_file)
    trained_accuracy = read_pickel(trained_accuracy_file)
    untrained_accuracy = read_pickel(untrained_accuracy_file)
    trained_loss = read_pickel(trained_loss_file)
    untrained_loss = read_pickel(untrained_loss_file)

    plotCombined(trained_epochs, trained_accuracy, untrained_accuracy, plot_x_axis, plot_y_axis_accuracy,
                 combined_accuracy_plot_file_name, combined_accuracy_plot_name)
    plotCombined(trained_epochs, trained_loss, untrained_loss, plot_x_axis, plot_y_axis_loss,
                 testing_loss_plot_file_name, testing_loss_plot_name)

    test_idxs = read_pickel(test_idxs_file)
    # Confusion Matrix for un-trained dataset

    confusion_matrix_untrained_nodes = confusion_matrix([(e) for e in labels_not_selected_doc_index],
                                                        list(pred_labels[test_idxs].max(1)[1].numpy()))

    plotConfusionMatrix(confusion_matrix_untrained_nodes, untrained_confusion_matrix_file_name)
    testing_precision, testing_recall, testing_F1_measure = calculateMetrics(confusion_matrix_untrained_nodes)

    plotBarChart(features, testing_precision, testing_precision_plot_name,
                 testing_precision_plot_file_name, categories_for_classification)
    plotBarChart(features, testing_recall, testing_recall_plot_name,
                 testing_recall_plot_file_name, categories_for_classification)
    plotBarChart(features, testing_F1_measure, testing_f1score_plot_name,
                 testing_f1score_plot_file_name, categories_for_classification)

    # Confusion Matrix for Trained dataset
    confusion_matrix_trained_nodes = confusion_matrix([(e) for e in labels_selected_doc_index],
                                                      list(pred_labels[selected].max(1)[1].numpy()))
    plotConfusionMatrix(confusion_matrix_trained_nodes, trained_confusion_matrix_file_name)
    training_precision, training_recall, training_F1_measure = calculateMetrics(confusion_matrix_trained_nodes)
    plotBarChart(features, training_precision, training_precision_plot_name,
                 training_precision_plot_file_name, categories_for_classification)
    plotBarChart(features, training_recall, training_recall_plot_name, training_recall_plot_file_name,
                 categories_for_classification)
    plotBarChart(features, training_F1_measure, training_f1score_plot_name,
                 training_f1score_plot_file_name, categories_for_classification)
