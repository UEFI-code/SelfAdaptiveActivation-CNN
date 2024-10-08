import torch
from torch.utils.data import Dataset
import cv2
import os
from config import MyConfigs
from torchvision import transforms
import numpy as np
from tqdm import tqdm

def get_files(file_dir,ratio):
    Images = []
    Labels = []
    for i in range(len(MyConfigs.classes)):
        for file in os.listdir(file_dir + MyConfigs.classes[i]):
            Images.append(file_dir + MyConfigs.classes[i] + '/' + file)
            Labels.append(i)
    temp = np.array([Images, Labels]).transpose()
    np.random.shuffle(temp)
    n_test = int(len(temp) * ratio)
    test_data = temp[0:n_test,:]
    train_data = temp[n_test:-1,:]
    return test_data,train_data

class datasets(Dataset):
    def __init__(self,data, transform = None, bigRAM = True):
        self.data = data.transpose()
        self.len = len(data)
        self.transform = transform
        self.imgs_path = self.data[0]
        self.labels = self.data[1]
        self.bigRAM = bigRAM
        if self.bigRAM:
            print('Wow Big RAM')
            self.imgs_cache = []
            for img_path in tqdm(self.imgs_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (MyConfigs.img_width, MyConfigs.img_height))
                img = torch.tensor(img).permute(2,0,1) / 255.0
                self.imgs_cache.append(img)
        
    def __getitem__(self,index):
        label = int(self.labels[index])
        if self.bigRAM:
            img = self.imgs_cache[index]
        else:
            img_path = self.imgs_path[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(MyConfigs.img_width,MyConfigs.img_height))
            img = torch.tensor(img).permute(2,0,1) / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return self.len

if __name__ == '__main__':
    test_data,_ = get_files(MyConfigs.data_folder,0.02)
    for i in (test_data):
        print(i)
    print(len(test_data))
    transforms = transforms.Compose([transforms.ColorJitter(0.05,0.05,0.05),])
    data = datasets(test_data, transform = transforms)
    test_item = data[0]
    print(f'tensor: {test_item[0]}, label: {MyConfigs.classes[test_item[1]]}')
