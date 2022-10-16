import torch
from torch import nn, optim
from calculate import writeTextFile, accuracy, save_as_pickle
from resources.constants import learning_rate, num_of_epochs, model_filename, regularization_factor, result_file, \
    X_input, Ahat_input, selected_index_file, selected_label_doc_index_file, not_selected_file, \
    not_selected_label_doc_index_file, loss_per_epochs_file, accuracy_per_epochs_file, trained_epochs_file, \
    trained_accuracy_file, untrained_accuracy_file, trained_loss_file, untrained_loss_file
from gcnmodel import gcn
from calculate import read_pickel
import time

if __name__ == '__main__':
    # Step 8. Graph Convolutional Network Model

    X = read_pickel(X_input)
    A_hat = read_pickel(Ahat_input)

    model = gcn(X.shape[1], A_hat)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000, 4000, 5000, 6000], gamma=0.77)
    loss_fun = torch.nn.CrossEntropyLoss()
    test_loss = []
    test_accuracy = []
    test_epochs = []
    loss_per_epochs = []
    accuracy_per_epochs = []
    trained_accuracy = []
    untrained_accuracy = []
    trained_epochs = []
    untrained_loss = []
    trained_loss = []
    l1_crit = nn.L1Loss(size_average=False)
    model.train()
    print('Learning Rate:', learning_rate)

    selected = read_pickel(selected_index_file)
    labels_selected_doc_index = read_pickel(selected_label_doc_index_file)
    not_selected = read_pickel(not_selected_file)
    labels_not_selected_doc_index = read_pickel(not_selected_label_doc_index_file)

    start_time = time.time()
    for epoch in range(num_of_epochs):
        optimizer.zero_grad()
        output = model(X)
        reg_loss = 0
        loss_train = loss_fun(output[selected], torch.tensor(labels_selected_doc_index))
        for param in model.parameters():
            reg_loss += l1_crit(param, target=torch.zeros_like(param))
        loss_train += regularization_factor * reg_loss
        loss_per_epochs.append(loss_train.item())
        training_accuracy, actual_labels, prediction = accuracy(output[selected], labels_selected_doc_index,
                                                                'Training Accuracy\n')
        accuracy_per_epochs.append(training_accuracy.item())
        loss_train.backward()
        optimizer.step()
        print('Epoch:' + str(epoch) + '\ttraining loss:' + str(loss_train.item()) +
              '\t training accuracy:' + str(training_accuracy.item()))
        '''if epoch % 5 == 0:
            test_epochs.append(epoch)
            test_output = model(X)
            loss_test = loss_fun(test_output[not_selected], torch.tensor(labels_not_selected))
            test_loss.append(loss_test.item())
            accuracy_test = accuracy(test_output[not_selected], labels_not_selected)
            test_accuracy.append(accuracy_test.item())
            print('Epoch:' + str(epoch) + '\tTesting loss:' + str(loss_test.item()) +
                  '\t Testing accuracy:' + str(accuracy_test.item()))'''

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                predicted_labels = model(X)
                trained_acc, training_actual_labels, training_prediction = accuracy(output[selected],
                                                                                    labels_selected_doc_index,
                                                                                    'Trained Accuracy\n')
                untrained_acc, testing_actual_labels, testing_prediction = accuracy(predicted_labels[not_selected],
                                                                                    labels_not_selected_doc_index,
                                                                                    'Un-Trained Accuracy\n')
                trained_accuracy.append(trained_acc.item())
                trainedLoss = loss_fun(output[selected], torch.tensor(labels_selected_doc_index))
                for param in model.parameters():
                    reg_loss += l1_crit(param, target=torch.zeros_like(param))
                trainedLoss += regularization_factor * reg_loss
                untrained_accuracy.append(untrained_acc.item())
                trained_epochs.append(epoch)
                trained_loss.append(trainedLoss.item())
                untrainedLoss = loss_fun(predicted_labels[not_selected], torch.tensor(labels_not_selected_doc_index))
                untrained_loss.append(untrainedLoss.item())
                model.train()
        scheduler.step()

    print("--- %s seconds ---" % (time.time() - start_time))
    # save the model to disk
    save_as_pickle(model_filename, model)
    save_as_pickle(loss_per_epochs_file, loss_per_epochs)
    save_as_pickle(accuracy_per_epochs_file, accuracy_per_epochs)
    save_as_pickle(trained_epochs_file, trained_epochs)
    save_as_pickle(trained_accuracy_file, trained_accuracy)
    save_as_pickle(untrained_accuracy_file, untrained_accuracy)
    save_as_pickle(trained_loss_file, trained_loss)
    save_as_pickle(untrained_loss_file, untrained_loss)

    # Write result on text file
    writeTextFile(result_file, "Training Loss\n", loss_per_epochs)
    writeTextFile(result_file, "Training Accuracy\n", accuracy_per_epochs)
    writeTextFile(result_file, "Testing Loss\n", test_loss)
    writeTextFile(result_file, "Testing Accuracy\n", test_accuracy)
    writeTextFile(result_file, "Trained Accuracy\n", trained_accuracy)
    writeTextFile(result_file, "Un-trained Accuracy\n", untrained_accuracy)
    writeTextFile(result_file, "Trained Loss\n", trained_loss)
    writeTextFile(result_file, "Un-trained Loss\n", untrained_loss)
