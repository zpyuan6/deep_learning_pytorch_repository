from cmath import nan
import torch
from utils.load_dataset import load_dataset
from utils.load_deep_learning_model import load_model
from utils.load_loss import load_loss_function
from utils.save_loss import LossHistory
from utils.utils import get_lr
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# model_path = 'saved_model\MRINet-3_classes-[128, 128, 128]_input_shape-ep020-loss0.395-val_loss1.508.pth'
# Cuda = True
# model_save_path = "saved_model"
# save_period = 5
# train_batch_size = 5
# val_batch_size = 1
# num_epochs = 100
# learning_rate=1e-4
# optimizer_name = "Adam"
# input_shape = [128, 128, 128]
# model_name = "MRINet"
# loss_name = "CrossEntropyLoss"
# # loss_name = "FocalLoss"
# parameters_and_grad_saved_path = f"saved_model/{model_name}_parameter_and_grad.txt"
# classes = ["CN","MCI","AD"]
# # classes = ["CN", "AD"]
# is_weight_loss = True

model_path = ''
Cuda = True
model_save_path = "saved_model"
save_period = 5
train_batch_size = 5
val_batch_size = 2
num_epochs = 100
learning_rate=1e-4
optimizer_name = "Adam"
input_shape = [128, 128, 128]
model_name = "Res3DNet34"
loss_name = "CrossEntropyLoss"
# loss_name = "FocalLoss"
parameters_and_grad_saved_path = f"saved_model/{model_name}_parameter_and_grad.txt"
classes = ["CN","MCI","AD"]
# classes = ["CN", "AD"]
is_weight_loss = True

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')

    # load data
    print("load data set")
    train_dataset,val_dataset,test_dataset = load_dataset(model_name = model_name, classes=classes, input_shape=input_shape)

    # 未设置共享内存 pin_memory：未设置 torch.utils.data.DataLoader 方法的 pin_memory 或者设置成 False,则数据需从 CPU 传入到缓存 RAM 里面，再给传输到 GPU 上, 优化：如果内存比较富裕，可以设置 pin_memory=True，直接将数据映射到 GPU 的相关内存块上，省掉一点数据传输时间
    # 多进程并行读取数据 设置 num_workers 等参数或者设置的不合理，会导致 cpu 性能没有跑起来，从而成为瓶颈，卡住 GPU
    # 未启用提前加载机制来实现 CPU 和 GPU 的并行 设置 prefetch_factor 等参数或者设置的不合理，导致 CPU 与 GPU 在时间上串行，CPU 运行时 GPU 利用率直接掉 0
    train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory = True, prefetch_factor=train_batch_size*2)
    val_dataset_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=4, pin_memory = True, prefetch_factor=val_batch_size*2)

    s,val_category = val_dataset.static_category()
    print("val_dataset: ",s)
    s,v = train_dataset.static_category()
    print("train_dataset: ",s)

    #  load model
    print("load deep learning model")
    model = load_model(model_name = model_name, classes=classes, input_shape=input_shape).cpu()
    # print(model)

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
    else:
        #  init parameter
        for m in model.modules():
            if isinstance(m, (nn.Conv2d,nn.Conv3d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))

    # load loss function
    print("load loss function")
    # create loss category weight
    # First method
    # category_num = train_dataset.static_category()[1]
    # num_of_samples = train_dataset.__len__()
    # weight = [num_of_samples/i for i in category_num]
    # Second method
    weight = [1.0 for i in range(len(classes))]
    if is_weight_loss:
        category_num = train_dataset.static_category()[1]
        max_num_in_category = max(category_num)
        weight = [max_num_in_category/i for i in category_num]
    print("weight: ",weight)
    loss_function = load_loss_function(loss_name=loss_name, weight=torch.tensor(weight))

    print("load loss history")
    loss_history = LossHistory(log_dir="loss_record", model=model, input_shape=input_shape)

    model_train = model.train()

    if Cuda:
        #multi GPU training
        model_train = torch.nn.DataParallel(model)
        # 设置为True，会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法。
        cudnn.benchmark = True
        # # 模型转移至cuda
        # model_train = model_train.cuda()
# ,weight_decay=0.01
    optimizer_mapping = {
        "Adam":optim.Adam(model_train.parameters(), lr=learning_rate),
        "SGD":optim.SGD(model_train.parameters(), lr=learning_rate),
        "RMSprop":optim.RMSprop(model_train.parameters(), lr=learning_rate)}
    
    optimizer = optimizer_mapping[optimizer_name]
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # Learning Scheduler
    
    epochs = range(num_epochs)
    print("Start epoch")
    train_step_num = int(train_dataset.__len__()/train_batch_size)
    val_step_num = int(val_dataset.__len__()/val_batch_size)
    print(f"train_step_num {train_step_num},val_step_num {val_step_num}")

    training_log = open("training.log","w")

    for epoch in epochs:
        loss        = 0
        val_loss    = 0
        loss_function = loss_function.to(device)
        model_train = model_train.to(device)

        print(f"Start {epoch} training epoch")
        train_true_sample={}
        train_prediction_sample={}
        num_true_sample = 0
        with tqdm(total=train_step_num,desc=f'Epoch {epoch + 1}/{num_epochs}',postfix=dict,mininterval=0.3) as pbar:
            for iteration,batch in enumerate(train_dataset_loader):
                mri, label, path = batch[0],batch[1],batch[2]
                # print("\n",mri[0].shape,torch.mean(mri[0]),torch.max(mri[0]),torch.min(mri[0]))
                # img = mri.cpu()
                # print(img.shape)

                # for i in range(1,11):
                #     plt.subplot(2,5,i)
                #     if i == 3:
                #         plt.title(path)
                #     plt.imshow(img[0].squeeze()[i*20,:,:])
                # plt.savefig(f"{iteration}-{label}.jpg")
                # print("mri",mri.shape)
                # print("label",label.shape)
                # print(torch.mean(mri),torch.std(mri),torch.max(mri),torch.min(mri))
                # print(label)
                # print(mri_path)
                
                # plt.imshow(mri.squeeze(0).squeeze(0)[:,78,:])
                # plt.show()

                if torch.max(torch.isnan(mri))==1:
                    raise Exception("data problem")

                with torch.no_grad():
                    mri = mri.to(device=device,dtype=torch.float)
                    label = label.to(device=device)
                
                optimizer.zero_grad()

                outputs = model_train(mri)
                # print("output: ",outputs)

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

                # print("outputs: ",outputs)
                # print("preds log: ",torch.log_softmax(outputs,1))
                # print("preds: ",preds)
                # print("label: ",label)
                loss_value = loss_function(outputs, label)

                # print("loss_value: ",loss_value)

                loss_value.backward()

                
                for name,parms in model_train.named_parameters():
                    # print(f"name: {name}, is_training: {parms.requires_grad}, value: {parms}, grad: {parms.grad}")
                    if torch.max(torch.isnan(parms.grad)):
                        raise Exception("Gradient problem")
                    # if parms.requires_grad:
                    #     training_log.writelines(f"{name}.grad: {parms.grad.mean()}:{parms.grad.min()}:{parms.grad.max()}")
                    #     # print("weight.grad: ", parms.grad.mean(), parms.grad.min(), parms.grad.max())
                    # break

                # update model parameters
                optimizer.step()

                # for name,parms in model_train.named_parameters():
                #     print(f"name: {name}, value: {parms}")
                #     break

                loss +=loss_value.item()
                # print(f'loss {loss},loss_value.item() {loss_value.item()},iteration {iteration+1}')

                pbar.set_postfix(**{'loss'  : '%.5f' %(loss / (iteration + 1)), 
                                'lr'    : get_lr(optimizer),
                                'acc' : '%.5f' %(num_true_sample / ((iteration + 1)*train_batch_size))})
                pbar.update(1)

        print(f"Prediction samples: {train_prediction_sample}|| True samples: {train_true_sample}")

        # training_log.writelines(f"-------------------Epoch {epoch}----------------------\n")
        # l = True
        # last_parms = {}
        for name,parms in model_train.named_parameters():
            loss_history.writer.add_histogram(name, parms.clone().cpu().data.numpy(), epoch)
            loss_history.writer.add_histogram(name+'/grad', parms.grad.clone().cpu().data.numpy(), epoch)
        #     if parms.requires_grad:
        #         if "weight" in name:
        #             last_parms = [name, parms]

        #         if l:
        #             training_log.writelines(f"{name}.grad: {parms.grad.mean()}:{parms.grad.min()}:{parms.grad.max()}\n")
        #             l = False
        
        # training_log.writelines(f"{last_parms[0]}.grad: {last_parms[1].grad.mean()}:{last_parms[1].grad.min()}:{last_parms[1].grad.max()}\n")
        exp_lr_scheduler.step()
        # continue

        print("Finish one training epoch")
        print("Start one val epoch")
        model_train.eval()
        true_sample={}
        prediction_sample={}
        num_true_sample = 0
        with tqdm(total=val_step_num, desc=f'Epoch {epoch + 1}/{num_epochs}',postfix=dict,mininterval=0.3) as pbar:
            for iteration, batch in enumerate(val_dataset_loader):
                mri, label = batch[0],batch[1]
                # 被该语句 wrap 起来的部分将不会track 梯度
                with torch.no_grad():
                    mri = mri.to(device=device,dtype=torch.float)
                    label = label.to(device=device)

                    optimizer.zero_grad()

                    outputs = model_train(mri)
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

        print("Finish one val epoch")

        training_loss = loss / train_step_num
        validation_loss = val_loss / val_step_num
        loss_history.append_loss(epoch + 1, training_loss, validation_loss)
        accuracy = num_true_sample/val_dataset.__len__()
        print('Epoch:'+ str(epoch + 1) + '/' + str(num_epochs))
        print(f'Total Samples: {val_dataset.__len__()} || Prediction samples: {prediction_sample}|| True samples: {num_true_sample} {true_sample} || Category: {val_category} ')
        print('Total Loss: %.5f || Val Loss: %.5f || Accuracy: %.5f ' % (training_loss, validation_loss, accuracy))

        if (epoch+1)%save_period==0 or epoch+1 == num_epochs:
            # Gpu save model will cause KeyError: 'module.conv.conv0_s1.weight' while load pth file
            torch.save(model.state_dict(), model_save_path+'/%s-%s_classes-%s_input_shape-ep%03d-loss%.3f-val_loss%.3f.pth'% (model_name, str(len(classes)),str(input_shape), epoch + 1, training_loss, validation_loss))

    training_log.close()

if __name__ == "__main__":
    train_model()