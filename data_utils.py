import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
    
class DRIVE_dataset(Dataset):
    def __init__(self, split='train', new_shape=(224, 224)):
        super(DRIVE_dataset, self).__init__()
        self.split = split

        self.path = '/home/anubhav/DRIVE_segmentation/DRIVE/'
        self.image_file_names = os.listdir(self.path + self.split + '/images/')

        self.transforms = transforms.Compose(
            [
                transforms.Resize(new_shape),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, index):
        if index >= len(self.image_file_names):
            print("Out of range")
            raise NotImplementedError
        
        image_name = self.image_file_names[index]
        manual_name = image_name[:3] + 'manual1.gif'
        
        image_path = self.path + self.split + '/images/' + image_name
        manual_path = self.path + self.split + '/1st_manual/' + manual_name
        
        image = Image.open(image_path)
        manual = Image.open(manual_path)

        image = self.transforms(image)
        manual = self.transforms(manual)

        return image, manual

if __name__ == '__main__':
    
    # drive = DRIVE_dataset(split='train')
    # img, label = drive[0]
    # print(img.shape, label.shape)
    train_dl = DataLoader(DRIVE_dataset(split='train'), batch_size=1, shuffle=True)
    for img, label in train_dl:
        print(img.shape, label.shape)
    img, label = next(iter(train_dl))
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img[0].permute(1,2,0))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(label[0].permute(1,2,0))
    plt.savefig('data.png')

    # train_dataloader = DataLoader(drive, batch_size=3, shuffle=True)

    # for i, (img, label) in enumerate(train_dataloader):
    #     print(i, ' ', img.shape, ' ', label.shape)