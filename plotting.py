import torch
from sklearn.metrics import confusion_matrix

from utils import plotGraph, plotCombined, labelDistribution, \
    plotConfusionMatrix, plotScatter, calculateMetrics, plotBarChart, read_pickel
from resources.constants import training_accuracy_plot_name, training_loss_plot_name, training_loss_plot_file_name, \
    training_accuracy_plot_file_name, plot_x_axis, plot_y_axis_loss, \
    plot_y_axis_accuracy, num_of_epochs, testing_loss_plot_file_name, testing_loss_plot_name, \
    combined_accuracy_plot_file_name, combined_accuracy_plot_name, untrained_confusion_matrix_file_name, \
    trained_confusion_matrix_file_name, training_precision_plot_file_name, \
    testing_precision_plot_file_name, training_recall_plot_file_name, testing_recall_plot_file_name, \
    training_f1score_plot_file_name, testing_f1score_plot_file_name, training_precision_plot_name, \
    training_recall_plot_name, training_f1score_plot_name, testing_precision_plot_name, testing_recall_plot_name, \
    testing_f1score_plot_name, \
    first_layer_before_training_for_testing_dataset_name, first_layer_before_training_for_training_dataset_file_name, \
    first_layer_before_training_for_testing_dataset_file_name, first_layer_before_training_for_training_dataset_name, \
    first_layer_after_training_for_training_dataset_name, first_layer_after_training_for_training_dataset_file_name, \
    first_layer_after_training_for_testing_dataset_name, first_layer_after_training_for_testing_dataset_file_name, \
    categories_for_classification, selected_index_file, selected_label_doc_index_file, not_selected_file, \
    not_selected_label_doc_index_file, model_filename, X_input, not_selected_label_file, selected_label_file, \
    test_idxs_file, features_file, loss_per_epochs_file, accuracy_per_epochs_file, trained_epochs_file, \
    trained_accuracy_file, untrained_accuracy_file, trained_loss_file, untrained_loss_file

if __name__ == '__main__':
    selected = read_pickel(selected_index_file)
    labels_selected_doc_index = read_pickel(selected_label_doc_index_file)
    not_selected = read_pickel(not_selected_file)
    labels_not_selected_doc_index = read_pickel(not_selected_label_doc_index_file)
    labels_not_selected = read_pickel(not_selected_label_file)
    labels_selected = read_pickel(selected_label_file)

    test_idxs = read_pickel(test_idxs_file)
    features = read_pickel(features_file)

    loss_per_epochs = read_pickel(loss_per_epochs_file)
    accuracy_per_epochs = read_pickel(accuracy_per_epochs_file)

    trained_epochs = read_pickel(trained_epochs_file)
    trained_accuracy = read_pickel(trained_accuracy_file)
    untrained_accuracy = read_pickel(untrained_accuracy_file)
    trained_loss = read_pickel(trained_loss_file)
    untrained_loss = read_pickel(untrained_loss_file)

    model = read_pickel(model_filename)
    X = read_pickel(X_input)

    plotScatter(model.weight, selected, labels_selected_doc_index, not_selected, labels_not_selected_doc_index,
                first_layer_before_training_for_training_dataset_name,
                first_layer_before_training_for_testing_dataset_name,
                first_layer_before_training_for_training_dataset_file_name,
                first_layer_before_training_for_testing_dataset_file_name)

    # Plot Loss and Accuracy
    plotGraph(range(num_of_epochs), loss_per_epochs, plot_x_axis, plot_y_axis_loss, training_loss_plot_file_name,
              training_loss_plot_name)
    plotGraph(range(num_of_epochs), accuracy_per_epochs, plot_x_axis, plot_y_axis_accuracy,
              training_accuracy_plot_file_name, training_accuracy_plot_name)

    '''plotGraph(test_epochs, test_loss, plot_x_axis, plot_y_axis_loss, testing_loss_plot_file_name,
              testing_loss_plot_name)
    plotGraph(test_epochs, test_accuracy, plot_x_axis, plot_y_axis_accuracy, testing_accuracy_plot_file_name,
              testing_accuracy_plot_name)'''

    plotCombined(trained_epochs, trained_accuracy, untrained_accuracy, plot_x_axis, plot_y_axis_accuracy,
                 combined_accuracy_plot_file_name, combined_accuracy_plot_name)
    plotCombined(trained_epochs, trained_loss, untrained_loss, plot_x_axis, plot_y_axis_loss,
                 testing_loss_plot_file_name, testing_loss_plot_name)

    # label distribution
    labelDistribution(labels_not_selected, labels_selected)

    # Confusion Matrix for un-trained dataset
    model.eval()
    with torch.no_grad():
        pred_labels = model(X)
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

    plotScatter(model.weight, selected, labels_selected_doc_index, not_selected, labels_not_selected_doc_index,
                first_layer_after_training_for_training_dataset_name,
                first_layer_after_training_for_testing_dataset_name,
                first_layer_after_training_for_training_dataset_file_name,
                first_layer_after_training_for_testing_dataset_file_name)

    ## TODO: Plot scatter graph for weights.
    '''plotScatter(model.weight2, selected, labels_selected, not_selected, labels_not_selected,
                'Classification of Trained Layer 2 weights','Classification of Un-trained Layer 2 weights',
                'Training Classification Layer 2 features.png', 'Testing Classification Layer 2 features.png')'''
