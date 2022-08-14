from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(
    model:torch.Module,
    device:torch.device,
    train_dataset:Dataset,
    train_dataset_loader:DataLoader,
    optimizer:optim.Optimizer,
    loss_function,
    train_batch_size:int,
    num_epochs:int,
    epoch:int):

    train_step_num = int(train_dataset.__len__()/train_batch_size)

    model_train = model.train()
    model_train.to(device)
    loss_function.to(device)

    train_prediction_sample = {}
    train_true_sample = {}
    num_true_sample = 0
    loss = 0

    with tqdm(total=train_step_num,desc=f'Epoch {epoch + 1}/{num_epochs}',postfix=dict,mininterval=0.3) as pbar:
        for iteration,batch in enumerate(train_dataset_loader):
            input, label, path = batch[0],batch[1],batch[2]

            if torch.max(torch.isnan(input))==1:
                raise Exception("data problem")

            with torch.no_grad():
                input = input.to(device=device,dtype=torch.float)
                label = label.to(device=device)
                
            optimizer.zero_grad()

            outputs = model_train(input)

            preds = torch.softmax(outputs,1)

            for index in range(len(preds)):
                    if int(torch.argmax(preds[index])) in train_prediction_sample:
                        train_prediction_sample[int(torch.argmax(preds[index]))]+=1
                    else: 
                        train_prediction_sample[int(torch.argmax(preds[index]))]=1

                    if torch.argmax(preds[index]) == label[index]:
                        num_true_sample += 1
                        if int(label[index]) in train_true_sample:
                            train_true_sample[int(label[index])]+=1
                        else: 
                            train_true_sample[int(label[index])]=1

            loss_value = loss_function(outputs, label)

            loss_value.backward()

            for name,parms in model_train.named_parameters():
                    if torch.max(torch.isnan(parms.grad)):
                        raise Exception("Gradient problem")

                # update model parameters
            optimizer.step()

            loss += loss_value.item()
                # print(f'loss {loss},loss_value.item() {loss_value.item()},iteration {iteration+1}')

            pbar.set_postfix(**{'loss'  : '%.5f' %(loss / (iteration + 1)), 
                                'lr'    : get_lr(optimizer),
                                'acc' : '%.5f' %(num_true_sample / ((iteration + 1)*train_batch_size))})
            pbar.update(1)

    print(f"Prediction samples: {train_prediction_sample}|| True samples: {train_true_sample}")


def val_model(
    model:torch.Module,
    device:torch.device,
    val_dataset:Dataset,
    val_dataset_loader:DataLoader,
    optimizer:optim.Optimizer,
    loss_function,
    val_batch_size:int,
    num_epochs:int,
    epoch:int):

    val_step_num = int(val_dataset.__len__()/val_batch_size)
    prediction_sample = {}
    true_sample = {}
    num_true_sample = 0
    val_loss = 0

    model_val = model.eval()
    model_val.to(device)
    loss_function.to(device)

    with tqdm(total=val_step_num, desc=f'Epoch {epoch + 1}/{num_epochs}',postfix=dict,mininterval=0.3) as pbar:
            for iteration, batch in enumerate(val_dataset_loader):
                mri, label = batch[0],batch[1]
                # 被该语句 wrap 起来的部分将不会track 梯度
                with torch.no_grad():
                    mri = mri.to(device=device,dtype=torch.float)
                    label = label.to(device=device)

                    optimizer.zero_grad()

                    outputs = model_val(mri)
                    preds = torch.softmax(outputs,1)

                    for index in range(len(preds)):
                        if int(torch.argmax(preds[index])) in prediction_sample:
                            prediction_sample[int(torch.argmax(preds[index]))]+=1
                        else: 
                            prediction_sample[int(torch.argmax(preds[index]))]=1

                        if torch.argmax(preds[index]) == label[index]:
                            # print(f"{ torch.argmax(preds[index])}:{label[index]}")
                            num_true_sample += 1
                            if int(label[index]) in true_sample:
                                true_sample[int(label[index])]+=1
                            else: 
                                true_sample[int(label[index])]=1

                    loss_value = loss_function(outputs, label)

                val_loss += loss_value.item()
                pbar.set_postfix(**{'val_loss': '%.5f' %(val_loss / (iteration + 1)), 'accuracy':'%.5f' %(num_true_sample / (iteration + 1)/ val_batch_size)})
                pbar.update(1)