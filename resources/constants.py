input_folder = 'paper_560/'

output_folder = './input_560_1'

output_file = './output_560_1/output.txt'
dataset_details = './output_560_1/DatasetDetails.csv'
text_graph_file_name = './output_560_1/Text Graph'
text_graph_name = 'Text Graph'
training_loss_plot_file_name = './output_560_1/Training Loss per epochs'
training_loss_plot_name = 'Training Loss per epochs'
training_accuracy_plot_file_name = './output_560_1/Training Accuracy per epochs'
training_precision_plot_file_name = './output_560_1/Precision Trained dataset'
training_recall_plot_file_name = './output_560_1/Recall Trained dataset'
training_f1score_plot_file_name = './output_560_1/F1 score of Trained dataset'
training_accuracy_plot_name = 'Training Accuracy per epochs'
training_precision_plot_name = 'Precision of Trained Dataset'
training_recall_plot_name = 'Recall of Trained Dataset'
training_f1score_plot_name = 'F1 score of Trained Dataset'
model_filename = 'finalized_model.pkl'
neural_ntw_details = './output_560_1/neural_ntw_details.csv'

testing_precision_plot_file_name = './output_560_1/Precision untrained dataset'
testing_recall_plot_file_name = './output_560_1/Recall untrained dataset'
testing_f1score_plot_file_name = './output_560_1/F1 score untrained dataset'
testing_precision_plot_name = 'Precision of Untrained Dataset'
testing_recall_plot_name = 'Recall of Untrained Dataset'
testing_f1score_plot_name = 'F1 score of Untrained Dataset'

testing_loss_plot_file_name = './output_560_1/Testing Loss per epochs'
testing_loss_plot_name = 'Testing Loss per epochs'
testing_accuracy_plot_file_name = './output_560_1/Testing Accuracy per epochs'
testing_accuracy_plot_name = 'Testing Accuracy per epochs'

combined_accuracy_plot_file_name = './output_560_1/Combined Accuracy per epochs'
combined_accuracy_plot_name = 'Trained and Un-trained Node accuracy'

tf_idf_histogram = './output_560_1/TF-IDF Histogram'
pmi_histogram = './output_560_1/PMI Histogram'
tf_idf_histogram_title = 'TF-IDF Histogram'
pmi_histogram_title = 'PMI Histogram'

first_layer_before_training_for_training_dataset_name = 'Classification of First Trained Layer weights Before Training'
first_layer_before_training_for_testing_dataset_name = 'Classification of First Un-trained Layer weights Before Training'
first_layer_before_training_for_training_dataset_file_name = './output_560_1/Training Classification First Layer Before Training'
first_layer_before_training_for_testing_dataset_file_name = './output_560_1/Testing Classification First Layer Before Training'
first_layer_after_training_for_training_dataset_name = 'Classification of First Trained Layer weights After Training'
first_layer_after_training_for_testing_dataset_name = 'Classification of First Un-trained Layer weights After Training'
first_layer_after_training_for_training_dataset_file_name = './output_560_1/Classification of First Trained Layer weights After Training'
first_layer_after_training_for_testing_dataset_file_name = './output_560_1/Classification of First Un-trained Layer weights After Training'

categories_for_classification = 'Categories of documents'

graph_details = './output_560_1/GraphDetails.csv'

model_digram = './output_560_1/gcn_model.png'

data_index_distribution = './output_560_1/data_index_distribution'
label_not_selected_distribution = './output_560_1/label_not_selected_distribution'
trained_confusion_matrix_file_name = './output_560_1/trained_confusion_matrix'
untrained_confusion_matrix_file_name = './output_560_1/untrained_confusion_matrix'

result_file = './output_560_1/result.txt'
accuracy_file = './output_560_1/prediction.txt'

output_column_filename = 'File Name'
output_column_noOfWords = 'Number of Words in File'
output_column_content = 'Content'

summary_column_noOfFiles = 'No Of Files in Dataset'
summary_column_noOfUniqueWords = 'No of Unique Words in Dataset'
summary_column_uniqueWords = 'Unique Words'

summary_column_avgWordCount = 'Average Number of words in documents'

graph_document_nodes = 'Document Nodes'
graph_word_nodes = 'Word Nodes'

graph_no_document_nodes = 'No of document nodes'
graph_no_word_nodes = 'No of word nodes'
graph_no_nodes = 'Total No of nodes'

graph_document_edges = 'Document to word Edges'
graph_word_edges = 'Word to word Edges'

graph_no_word_edges = 'No of word edges'
graph_no_document_edges = 'No of document edges'
graph_no_edges = 'Total No of edges'

training_dataset_size = 'Size of Training Data'
testing_dataset_size = 'Size of Testing Data'

text_graph_pkl_file_name = 'text_graph2.pkl'
word_edge_graph_pkl_file_name = 'word_word_edges2.pkl'
test_index_file_name = 'test_idxs.pkl'
selected_index_file = 'selected.pkl'
not_selected_file = 'notselected.pkl'
selected_label_file = 'labels_selected.pkl'
not_selected_label_file = 'labels_not_selected.pkl'
selected_label_doc_index_file = 'labels_selected.pkl'
not_selected_label_doc_index_file = 'labels_not_selected.pkl'
X_input = 'X.pkl'
f_function = 'f.pkl'
Ahat_input = 'Ahat.pkl'
graph_input = 'graph.pkl'

selected_index_training = 'selected_index.pkl'
selected_label_doc_index = 'selected_label_doc_index.pkl'
not_selected_index_testing = 'not_selected_index_testing.pkl'
not_selected_label_doc_index_testing = 'not_selected_label_doc_index_testing.pkl'
test_idxs_file = 'test_idxs_file.pkl'
features_file = 'features.pkl'

loss_per_epochs_file = 'loss_per_epochs.pkl'
accuracy_per_epochs_file = 'accuracy_per_epochs.pkl'
trained_epochs_file = 'trained_epochs.pkl'
trained_accuracy_file = 'trained_accuracy.pkl'
untrained_accuracy_file = 'untrained_accuracy.pkl'
trained_loss_file = 'trained_loss.pkl'
untrained_loss_file = 'untrained_loss.pkl'

resource_path = 'resources/labels'

log_save_graph = 'Created Graph Saved...'
log_pkl_saved = 'Pkl file is already saved...'
log_add_doc_node = 'Adding document nodes to graph...'
log_building_graph = 'Building graph (No. of document, word nodes: %d, %d)...'
log_training_starts = 'Training Process starts...'

plot_x_axis = 'Epochs'
plot_y_axis_loss = 'Loss'
plot_y_axis_accuracy = 'Accuracy'
plot_y_axis_precision = 'Precision'
plot_y_axis_recall = 'Recall'
plot_y_axis_f1score = 'F1 Score'

trained_nodes = 'Trained Nodes'
untrained_nodes = 'Untrained Nodes'

'''
No of neurons in hidden layer = (Input size * 2/3) + no of output classes
'''
# hidden_layer_1_size = 330
# hidden_layer_2_size = 130 # 25
hidden_layer_1_size = 250
hidden_layer_2_size = 100  # 25
no_output_classes = 7
learning_rate = 0.0001
num_of_epochs = 1001
dropout = 0.50
weight_decay = 1e-5
regularization_factor = 0.000005
# regularization_factor = 1e-8
sliding_window_size = 5
