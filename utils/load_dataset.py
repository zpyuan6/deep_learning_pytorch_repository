from code import interact
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from dataset.AD_MRI_Dataset import MyADMRIDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from matplotlib.widgets import Slider


def load_dataset(input_shape, model_name, classes=["CN","MCI","AD"]):
    train_dataset_file_path = "dataset_file\\train.txt"
    # train_dataset_file_path = "dataset_file\\train_for_test.txt"
    val_dataset_file_path = "dataset_file\\val.txt"
    test_dataset_file_path = "dataset_file\\test.txt"


    train_dataset = MyADMRIDataset(dataset_file = train_dataset_file_path,  model=model_name, num_classes=len(classes), input_shape=input_shape)
    
    val_dataset = MyADMRIDataset(dataset_file = val_dataset_file_path, model=model_name, num_classes=len(classes), input_shape=input_shape)

    test_dataset = MyADMRIDataset(dataset_file = test_dataset_file_path, model=model_name, num_classes=len(classes), input_shape=input_shape)

    # present data in dataset
    # for img,label in dataset_loader:
    #     print(label)
    #     print(np.shape(img[0]))

    #     plt.imshow(img[0][:,100,:])
    #     plt.show()
    #     # interact_show_mri(img[0])
    #     break

    return train_dataset,val_dataset,test_dataset

def main():
    # test_file = ["F:\ADNI\ADNI1_Screening 1.5T\ADNI\\002_S_0295\MPR__GradWarp__B1_Correction__N3__Scaled\\2006-04-18_08_20_30.0\I45108\\ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070319113623975_S13408_I45108.nii","F:\ADNI\ADNI1_Screening 1.5T\ADNI\\002_S_0295\MPR__GradWarp__B1_Correction__N3__Scaled_2\\2006-04-18_08_20_30.0\I118671\\ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114556321_S13408_I118671.nii"]
    # show_nii(test_file)
    load_dataset()

    

if __name__ == "__main__":
    main()