from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import patchify
import numpy as np
import matplotlib.gridspec as gridspec
import glob as glob
import os
import cv2
import torch
from torch.utils.data import DataLoader, Dataset

#Before running make sure that you already have 2 folders: DIV2k for the training images and DIVK_VAL for the validation images.

STRIDE = 96
SIZE = 32

def create_patches(input_path,out_hr_path,out_lr_path2 , out_lr_path4):
    os.makedirs(out_hr_path, exist_ok=True)
    os.makedirs(out_lr_path2, exist_ok=True)
    os.makedirs(out_lr_path4, exist_ok=True)
    
    all_paths = []
    all_paths.extend(glob.glob(f"{input_path}/*"))
    print(f"Creating patches for {len(all_paths)} images")

    for image_path in tqdm(all_paths, total=len(all_paths)):
        image = Image.open(image_path)
        image_name = image_path.split(os.path.sep)[-1].split('.')[0]
        w, h = image.size
        # Create patches
        patches = patchify.patchify(np.array(image), (32, 32, 3), STRIDE)
        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = patches[i, j, 0, :, :, :]
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

                #HR patch
                cv2.imwrite(f"{out_hr_path}/{image_name}_{counter}.png",patch)
                h, w, _ = patch.shape

                #LR patch down-sampled x2
                low_res_img2 = cv2.resize(patch, (int(w/2), int(h/2)), interpolation=cv2.INTER_CUBIC)
                high_res_upscale2 = cv2.resize(low_res_img2, (w, h), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f"{out_lr_path2}/{image_name}_{counter}.png",high_res_upscale2)
                
                #LR patch down-sampled x4
                low_res_img4 = cv2.resize(patch, (int(w/4), int(h/4)), interpolation=cv2.INTER_CUBIC)
                high_res_upscale4 = cv2.resize(low_res_img4, (w, h), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f"{out_lr_path4}/{image_name}_{counter}.png",high_res_upscale4)

#Create the training set
create_patches('../input/DIV2K','../input/train_hr_patches','../input/train_lr_patches2','../input/train_lr_patches4')

#
images = []
path='../input/DIVK_VAL'
images.extend(glob.glob(f"{path}/*.png"))
print(len(images))

os.makedirs('../input/Val_2x', exist_ok=True)
save_path_lr2 = '../input/Val_2x'
os.makedirs('../input/Val_hr', exist_ok=True)
save_path_hr = '../input/Val_hr'
os.makedirs('../input/Val_4x', exist_ok=True)
save_path_lr4 = '../input/Val_4x'

print(f"High resolution images save path: {save_path_hr}")
print(f"Low resolution x2 images save path: {save_path_lr2}")
print(f"Low resolution x4 images save path: {save_path_lr4}")

for image in tqdm(images, total=len(images)):
    orig_img = Image.open(image)
    image_name = image.split(os.path.sep)[-1]
    w, h = orig_img.size[:]

    orig_img.save(f"{save_path_hr}/{image_name}")

    low_res_img2 = orig_img.resize((int(w*0.5), int(h*0.5)), Image.BICUBIC)
    high_res_upscale2 = low_res_img2.resize((w, h), Image.BICUBIC)
    high_res_upscale2.save(f"{save_path_lr2}/{image_name}")

    low_res_img4 = orig_img.resize((int(w*0.25), int(h*0.25)), Image.BICUBIC)
    high_res_upscale4 = low_res_img4.resize((w, h), Image.BICUBIC)
    high_res_upscale4.save(f"{save_path_lr4}/{image_name}")

#Creation of the dataloader class that will process the dataset before training

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1

class CNNDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.all_image_paths = glob.glob(f"{image_paths}/*")

        self.all_label_paths = glob.glob(f"{label_paths}/*") 

    def __len__(self):
        return (len(self.all_image_paths))
    
    def __getitem__(self, index):
        image = Image.open(self.all_image_paths[index]).convert('RGB')
        label = Image.open(self.all_label_paths[index]).convert('RGB')

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)

        image /= 255.
        label /= 255.

        image = image.transpose([2, 0, 1])
        label = label.transpose([2, 0, 1])

        return (torch.tensor(image, dtype=torch.float),torch.tensor(label, dtype=torch.float))

# Prepare the datasets.
def get_datasets(train_image_paths, train_label_paths,valid_image_path, valid_label_paths):
    dataset_train = CNNDataset(train_image_paths, train_label_paths)

    dataset_valid = CNNDataset( valid_image_path, valid_label_paths)
    
    return dataset_train, dataset_valid

# Prepare the data loaders
def get_dataloaders(dataset_train, dataset_valid):
    train_loader = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE,shuffle=True)
    
    valid_loader = DataLoader(dataset_valid, batch_size=TEST_BATCH_SIZE,shuffle=False)
    
    return train_loader, valid_loader