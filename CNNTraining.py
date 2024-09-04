import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import os
from torchvision.utils import save_image
import time
import CNNmodel
import os

from tqdm import tqdm
from datasets import get_datasets, get_dataloaders

def psnr(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    diff = outputs - label
    rmse = math.sqrt(np.mean((diff) ** 2))
    if rmse == 0:
        return 100 #Vert unlikely edge case
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR

def save_plot(train_loss, val_loss, train_psnr, val_psnr,counter):
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/'+str(counter)+'/loss.png')
    plt.close()
    plt.close('all')

    # PSNR plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig('../outputs/'+str(counter)+'/psnr.png')
    plt.close('all')

    #Saving the values of MSE and PSNR in order merge the plots if training is stopped and then resumed
    # with open("../outputs/"+str(counter)+"/Performance.txt", 'w') as f:
    #     for i in train_loss:
    #         f.write(str(i)+" ,")
    #     f.write("\n")
    #     for i in val_loss:
    #         f.write(str(i)+" ,")
    #     f.write("\n")
    #     for i in train_psnr:
    #         f.write(str(i)+" ,")
    #     f.write("\n")
    #     for i in val_psnr:
    #         f.write(str(i)+" ,")
    #     f.write("\n")


def save_model_state(model,counter):
    print('Saving model')
    torch.save(model.state_dict(), '../outputs/'+str(counter)+'/model.pth')

def save_model(epochs, model, optimizer, criterion, counter):
    torch.save({'epoch': epochs+1,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': criterion,}, f"../outputs/{counter}/model_ckpt.pth")



TRAIN_LABEL_PATHS = '../input/train_hr_patches'
TRAN_IMAGE_PATHS = '../input/train_lr_patches2'  
VALID_LABEL_PATHS = '../input/Val_hr'
VALID_IMAGE_PATHS = '../input/Val_2x'  



def train(model, dataloader):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        image_data = data[0].to(device)
        label = data[1].to(device)
        
        # Zero grad the optimizer.
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, label)

        # Backpropagation.
        loss.backward()
        # Update the parameters.
        optimizer.step()

        # Add loss of each item (total items in a batch = batch size).
        running_loss += loss.item()
        # Calculate batch psnr (once every `batch_size` iterations).
        batch_psnr =  psnr(label, outputs)
        running_psnr += batch_psnr

    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr

def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_data = data[0].to(device)
            label = data[1].to(device)
            
            outputs = model(image_data)
            loss = criterion(outputs, label)

            running_loss += loss.item()
 
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr

    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr


#Training of our models

hyper=[[0.001,[128,64],[9,1,5],10]  ,  [0.0001,[128,64],[9,1,5],10]  ,  [0.00001,[128,64],[9,1,5],10]  ,   [0.0001,[200,100],[9,1,5],4] ,   [0.0001,[64,32],[9,1,5],4]  ]

for i,param in enumerate(hyper):
    # Learning parameters.
    epochs = param[3] # Number of epochs 
    lr = param[0] # Learning rate.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model.
    print(i) #Check where it is saved
    print('Computation device: ', device)
    model = CNNmodel.CNN(param[1],param[2]).to(device)
    
    print(model)

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function. 
    criterion = nn.MSELoss()

    dataset_train, dataset_valid = get_datasets(TRAN_IMAGE_PATHS, TRAIN_LABEL_PATHS,VALID_IMAGE_PATHS, VALID_LABEL_PATHS)
    train_loader, valid_loader = get_dataloaders(dataset_train, dataset_valid)

    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_valid)}")

    os.makedirs('../outputs/'+str(i), exist_ok=True)
    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_psnr = train(model, train_loader)
        val_epoch_loss, val_epoch_psnr = validate(model, valid_loader)
        print(f"Train PSNR: {train_epoch_psnr:.3f}")
        print(f"Val PSNR: {val_epoch_psnr:.3f}")
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)
        
    save_model_state(model,i)
    save_plot(train_loss, val_loss, train_psnr, val_psnr,i)
    save_model(epoch, model, optimizer, criterion,i)

    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")


#Training on the 4x down-sampled
TRAN_IMAGE_PATHS = '../input/train_lr_patches4'  
VALID_IMAGE_PATHS = '../input/Val_4x'  
hyper=[[0.0001,[128,64],[9,1,5],10]  ,  [0.0001,[128,64],[9,1,5],10] ]

for i,param in enumerate(hyper):
    # Learning parameters.
    epochs = param[3] # Number of epochs 
    lr = param[0] # Learning rate.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if i==1:
        weight_path = '../outputs/' + str(1) + "/model_ckpt.pth"
        print('Loading weights to resume training...')
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    j=i+5 #To save the results somewhere else

    print('Computation device: ', device)
    model = CNNmodel.CNN(param[1],param[2]).to(device)
    
    print(model)

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function. 
    criterion = nn.MSELoss()

    dataset_train, dataset_valid = get_datasets(TRAN_IMAGE_PATHS, TRAIN_LABEL_PATHS,VALID_IMAGE_PATHS, VALID_LABEL_PATHS)
    train_loader, valid_loader = get_dataloaders(dataset_train, dataset_valid)

    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_valid)}")

    os.makedirs('../outputs/'+str(j), exist_ok=True)
    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_psnr = train(model, train_loader)
        val_epoch_loss, val_epoch_psnr = validate(model, valid_loader)
        print(f"Train PSNR: {train_epoch_psnr:.3f}")
        print(f"Val PSNR: {val_epoch_psnr:.3f}")
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)
        
    save_model_state(model,j)
    save_plot(train_loss, val_loss, train_psnr, val_psnrji)
    save_model(epoch, model, optimizer, criterion,j)

    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")


