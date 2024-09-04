import torch
import glob as glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from CNNmodel import CNN
from PIL import Image
from utils import psnr
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_data = data[0].to(device)
            label = data[1].to(device)
            outputs = model(image_data)
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr

    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr

# The SRCNN dataset module.
class CNNDataset(Dataset):
    def __init__(self, image_paths,scale):
        self.all_image_paths = glob.glob(f"{image_paths}/*")
        self.scale=scale

    def __len__(self):
        return (len(self.all_image_paths))

    def __getitem__(self, index):

        label = Image.open(self.all_image_paths[index]).convert('RGB')
        w, h = label.size[:]

        low_res_img = label.resize((int(w/self.scale), int(h/self.scale)), Image.BICUBIC)

        image = low_res_img.resize((w, h), Image.BICUBIC)

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)

        image /= 255.
        label /= 255.

        image = image.transpose([2, 0, 1])
        label = label.transpose([2, 0, 1])

        return (torch.tensor(image, dtype=torch.float),torch.tensor(label, dtype=torch.float))

# Prepare the datasets.
def get_datasets(image_paths,scale):
    dataset_test = CNNDataset(image_paths,scale)
    return dataset_test

# Prepare the data loaders
def get_dataloaders(dataset_test):
    test_loader = DataLoader(dataset_test, batch_size=1,shuffle=False)
    return test_loader


# Load the model.

model = CNN([128,64],[9,1,5]).to(device)
model.load_state_dict(torch.load('../outputs/1/model.pth'))

data_paths = [['../input/Set5/original', 'Set5'],['../input/Set14/original', 'Set14']]

for data_path in data_paths:
    dataset_test = get_datasets(data_path[0],2)
    test_loader = get_dataloaders(dataset_test)

    _, test_psnr = validate(model, test_loader, device)
    print(f"Test PSNR on {data_path[1]}: {test_psnr:.3f}")


model = CNN([128,64],[9,1,5]).to(device)
model.load_state_dict(torch.load('../outputs/6/model.pth'))

data_paths = [['../input/Set5/original', 'Set5'],['../input/Set14/original', 'Set14']]

for data_path in data_paths:
    dataset_test = get_datasets(data_path[0],4)
    test_loader = get_dataloaders(dataset_test)

    _, test_psnr = validate(model, test_loader, device)
    print(f"Test PSNR on {data_path[1]}: {test_psnr:.3f}")