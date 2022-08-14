from sklearn.metrics import accuracy_score
from utils.load_deep_learning_model import load_model
from utils.load_dataset import load_dataset
from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
import numpy as np

evaluation_file_path = "loss_record"
Cuda = True
model_path = 'saved_model\Res3DNet-2_classes-ep100-loss0.825-val_loss0.819.pth'
classes = ['CN','AD']
batch_size=5

def do_prediction():

    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')

    model = load_model(model_name,classes=classes)

    print('Load weights {}.'.format(model_path))
        # state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
    model_dict      = model.state_dict()
        # Loads an object saved with torch.save() from a file.
    pretrained_dict = torch.load(model_path, map_location = device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    train_dataset,val_dataset,test_dataset = load_dataset(model_name=model_name, classes=classes)

    test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory = True, prefetch_factor=batch_size*2)

    model_name = model_path.split("\\")[-1][0:-4]
    evaluation_file = open(os.path.join(evaluation_file_path,f'{model_name}_evaluation.txt'), 'w')

    test_samples_num = test_dataset.__len__()

    matric_map = {"TP":[0 for i in range(len(classes))], "TN":[0 for i in range(len(classes))], "FP":[0 for i in range(len(classes))], "FN":[0 for i in range(len(classes))]}

    print(matric_map)

    print("Start prediction")
    with tqdm(total=test_samples_num/batch_size,postfix=dict,mininterval=0.3) as pbar:
        for iteration,batch in enumerate(test_dataset_loader):
            mri, label = batch[0],batch[1]

            with torch.no_grad():
                mri = mri.to(device=device,dtype=torch.float)
                label = label.to(device=device,dtype=torch.float)
                model = model.to(device=device)

            model = model.eval()
                
            outputs = model(mri)
            preds = torch.softmax(outputs,1)

            for index in range(len(label)):
                content = "{}:{}:{}\n".format(torch.argmax(preds[index]),label[index],preds[index])
                # print("preds: ",preds[index])
                # print("label: ",label[index])
                evaluation_file.write(content)

                for i in range(len(classes)):
                    if i == label[index]:
                        if i == torch.argmax(preds[index]):
                            matric_map["TP"][i]+=1
                        else:
                            matric_map["FN"][i]+=1
                    else:
                        if i == torch.argmax(preds[index]):
                            matric_map["FP"][i]+=1
                        else:
                            matric_map["TN"][i]+=1

            pbar.update(1)

    print(matric_map)
    evaluation_file.write(str(matric_map))
    
    print("Finish prediction")

    accuracy = [0 for i in range(len(classes))]
    precision = [0 for i in range(len(classes))]
    recall = [0 for i in range(len(classes))]
    for i in range(len(classes)):
        accuracy[i] = (matric_map["TP"][i]+matric_map["TN"][i])/(matric_map["TP"][i]+matric_map["TN"][i]+matric_map["FP"][i]+matric_map["FN"][i])
        
        if (matric_map["TP"][i]+matric_map["FP"][i]) == 0:
            precision[i] = 0
        else:   
            precision[i] = (matric_map["TP"][i])/(matric_map["TP"][i]+matric_map["FP"][i])
            
        if (matric_map["TP"][i]+matric_map["FN"][i]) == 0:
            recall[i] = 0
        else:
            recall[i] = (matric_map["TP"][i])/(matric_map["TP"][i]+matric_map["FN"][i])

    evaluation_file.write("\n accuracy {}, \n precision {}, \n recall {}".format(accuracy,precision,recall))
    evaluation_file.close()



if __name__ == "__main__":
    do_prediction()

