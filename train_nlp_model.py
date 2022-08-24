import torch
from load_cyberbullying_data import load_data
from torch.utils.data import DataLoader
from model.BiLSTM import BiLSTM_Sentiment_Classifier
import numpy as np
import torch.nn as nn
from utils.training import train_model,val_model
from utils.save_loss import LossHistory

from sklearn.metrics import classification_report, confusion_matrix

from  torchsummary import summary

Cuda = True
BATCH_SIZE = 32
HIDDEN_DIM = 100 #number of neurons of the internal state (internal neural network in the LSTM)
LSTM_LAYERS = 1 #Number of stacked LSTM layers

LR = 3e-4 #Learning rate
DROPOUT = 0.5 #LSTM Dropout
BIDIRECTIONAL = True #Boolean value to choose if to use a bidirectional LSTM or not
EPOCHS = 5 #Number of training epoch
EMBEDDING_DIM = 200

model_save_path="saved_model"
model_path=''
save_period=1
model_name="bilstm"
classes=["not_cyberbullying","religion","age","ethnicity","gender","other_cyberbullying"]

def train_nlp_model():
    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')

    # load data
    train_data, val_data, test_data, VOCAB_SIZE = load_data()

    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True, num_workers=4, pin_memory = True, prefetch_factor=BATCH_SIZE*2) 
    val_loader = DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True, num_workers=4, pin_memory = True, prefetch_factor=BATCH_SIZE*2)

    #  load model
    print("load deep learning model")
    model = BiLSTM_Sentiment_Classifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM,len(classes), LSTM_LAYERS,BIDIRECTIONAL, BATCH_SIZE, DROPOUT, device)

    #Initialize embedding with the previously defined embedding matrix
    model.embedding.weight.data.copy_(torch.from_numpy(np.load("embedding_word.npy")))
    #Allow the embedding matrix to be fined tuned to better adapt to out dataset and get higher accuracy
    model.embedding.weight.requires_grad=True

        # load model file
    if model_path != '':
        print('Load weights {}'.format(model_path))
        # state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
        model_dict      = model.state_dict()
        # Loads an object saved with torch.save() from a file.
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        # for k, v in pretrained_dict.items():
        #     print("pretrained_dict",k,v)
        #     break
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # for k, v in model.named_parameters():
        #     print("model",k,v)
        #     break

    print(model)
    

    print("load loss function")

    loss_function = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay = 5e-6)

    train_step_num = int(train_data.__len__()/BATCH_SIZE)
    val_step_num = int(val_data.__len__()/BATCH_SIZE)

    loss_history = LossHistory(log_dir="loss_record", model=model, input_shape=[82])

    for epoch in range(EPOCHS):
        #lists to host the train and validation losses of every batch for each epoch
        train_loss, valid_loss  = [], []
        #lists to host the train and validation accuracy of every batch for each epoch
        train_acc, valid_acc  = [], []

        #lists to host the train and validation predictions of every batch for each epoch
        y_train_list, y_val_list = [], []

        #initalize number of total and correctly classified texts during training and validation
        correct, correct_val = 0, 0
        total, total_val = 0, 0
        running_loss, running_loss_val = 0, 0

        print(f"Start {epoch} training epoch")

        total_step = len(train_loader)
        total_step_val = len(val_loader)

        model.to(device)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) #load features and targets in device
            model.train()

            h = model.init_hidden(labels.size(0))

            model.zero_grad() #reset gradients 

            output, h = model(inputs,h) #get output and hidden states from LSTM network
            
            loss = loss_function(output, labels)
            loss.backward()
            
            running_loss += loss.item()
            
            optimizer.step()

            y_pred_train = torch.argmax(output, dim=1) #get tensor of predicted values on the training set
            y_train_list.extend(y_pred_train.squeeze().tolist()) #transform tensor to list and the values to the list
            
            correct += torch.sum(y_pred_train==labels).item() #count correctly classified texts per batch
            total += labels.size(0) #count total texts per batch

        train_loss.append(running_loss / total_step)
        train_acc.append(100 * correct / total)

        print("Finish one training epoch")
        print("Start one val epoch")

        with torch.no_grad():
        
            model.eval()
                
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                val_h = model.init_hidden(labels.size(0))

                output, val_h = model(inputs, val_h)

                val_loss = loss_function(output, labels)
                running_loss_val += val_loss.item()

                y_pred_val = torch.argmax(output, dim=1)
                y_val_list.extend(y_pred_val.squeeze().tolist())

                correct_val += torch.sum(y_pred_val==labels).item()
                total_val += labels.size(0)

            valid_loss.append(running_loss_val / total_step_val)
            valid_acc.append(100 * correct_val / total_val)

        print("Finish one val epoch")

        training_loss = loss / train_step_num
        validation_loss = val_loss / val_step_num
        loss_history.append_loss(epoch + 1, training_loss, validation_loss)
        print('Epoch:'+ str(epoch + 1) + '/' + str(EPOCHS))
        print('Total Loss: %.5f || Val Loss: %.5f' % (training_loss, validation_loss))

        if (epoch+1)%save_period==0 or epoch+1 == EPOCHS:
            # Gpu save model will cause KeyError: 'module.conv.conv0_s1.weight' while load pth file
            torch.save(model.state_dict(), model_save_path+'/%s-%s_classes-ep%03d-loss%.3f-val_loss%.3f.pth'% (model_name, str(len(classes)), epoch + 1, training_loss, validation_loss))

def evaluation_model():
    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')

    # load data
    train_data, val_data, test_data, VOCAB_SIZE = load_data()

    test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True, num_workers=4, pin_memory = True, prefetch_factor=BATCH_SIZE*2)


    model = BiLSTM_Sentiment_Classifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM,len(classes), LSTM_LAYERS,BIDIRECTIONAL, BATCH_SIZE, DROPOUT, device)

    # summary(model,(1,100))

    model.load_state_dict(torch.load("saved_model\\bilstm-6_classes-ep005-loss0.000-val_loss0.003.pth"))
    model.to(device)
    model.eval()
    y_pred_list = []
    y_test_list = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        test_h = model.init_hidden(labels.size(0))
        print(f"")
        output, val_h = model(inputs, test_h)
        print(f"input_shape: {inputs.shape}\n labels_shape: {labels.shape},")
        y_pred_test = torch.argmax(output, dim=1)
        y_pred_list.extend(y_pred_test.squeeze().tolist())
        y_test_list.extend(labels.squeeze().tolist())

        break

    print('Classification Report for Bi-LSTM :\n', classification_report(y_test_list, y_pred_list, target_names=classes))


if __name__ == "__main__":
    # train_nlp_model()
    evaluation_model()