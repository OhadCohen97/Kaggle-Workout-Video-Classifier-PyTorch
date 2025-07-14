import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from early_stopping_pytorch import EarlyStopping
from torch import optim
from torchinfo import summary
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from gymdata import GymDataset
from VideoEncoder import VideoModel
from CNN3D import CNNClassifayer



EPOCHS = 500
RANDOM_SEED = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


VIDEO_PATH = '/home/dsi/ohadico97/homework/data'
MP4_FILES = glob.glob(os.path.join(VIDEO_PATH, '**', '*.mp4'), recursive=True)


def prep_data(BATCH_SIZE=32):
    
    train_files, temp_files = train_test_split(MP4_FILES, test_size=100, random_state=RANDOM_SEED) # 200 samples out from train to val-test splits  --> 150 for val and 50 for test
    val_files, test_files = train_test_split(temp_files, test_size=50, random_state=RANDOM_SEED)

    train_dataset = GymDataset(train_files, augment=True)
    val_dataset = GymDataset(val_files, augment=False)
    test_dataset = GymDataset( test_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers = 16,pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=True,num_workers = 16,pin_memory=True,persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False,num_workers = 16,pin_memory=True,persistent_workers=True)

    return train_loader,val_loader,test_loader

def train_model(model, data_loader, optimizer,criterion,device):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        x,y = batch
        inputs = x.to(device)
        targets = y.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
            
        loss = criterion(outputs, targets) # for bilstm

        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    return total_loss / len(data_loader)


def validate(model, data_loader, criterion,device):
    """
    Runs validation.
    
    Args:
        model: Trained PyTorch model.
        dataloader: Validation DataLoader.
        device: CUDA or CPU.
    """
    total_loss = 0
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)  
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(targets.cpu().tolist())
    print("Val Accuracy: ",accuracy_score(actual_labels, predictions))

    return  total_loss / len(data_loader)


def main(model_name):

    if model_name == 'None':
        print("Fine-tuning with R(2+1)D")#patience=5
        BATCH_SIZE = 32
        LR = 0.0001
        es_patience = 5
        model = VideoModel(model_name=None) #R(2+1)D
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-5) 
    elif model_name == 'x3d_s':
        print("Fine-tuning with X3D_S") #patience=5
        BATCH_SIZE = 64
        LR = 0.0001
        es_patience = 5
        model = VideoModel(model_name=model_name)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-5) # x3d_s
    elif model_name == '3DCNN':
        print("Training with custom CNN")
        BATCH_SIZE = 16
        LR = 0.001
        es_patience = 20
        model = CNNClassifayer()
        optimizer = optim.Adam(model.parameters())



    TRAINED_MODEL_PATH = f"/home/dsi/ohadico97/homework/trained_{model_name}_{LR}_{BATCH_SIZE}.pth"

    model.to(device)
    train_data_l,val_data_l,test_data_l = prep_data(BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=es_patience, verbose=True,path=f"/home/dsi/ohadico97/homework/checkpoint_{model_name}_{LR}_{BATCH_SIZE}.pth")
    # print(summary(model, input_size=(BATCH_SIZE,1, 32, 224, 224)))
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_loss = train_model(model, train_data_l, optimizer,criterion, device)
       
        val_loss = validate(model, val_data_l, criterion, device)

        print(f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
    torch.save(model.state_dict(), TRAINED_MODEL_PATH)

    test(model,test_data_l,device)


def test(model, data_loader,device):
    """
    Test the model.
    
    Args:
        model: Trained PyTorch model.
        dataloader: Validation DataLoader.
        device: CUDA or CPU.
    """

    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)  
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(targets.cpu().tolist())

    print("Test Accuracy: ",accuracy_score(actual_labels, predictions))
    

if __name__ == "__main__":
    model_name = "3DCNN" # x3d_s |"None" | "3DCNN"

    # main(model_name)
    # ----- Test after training ----#
    test_model_after_training = True
    if test_model_after_training:
        # model = VideoModel(model_name=None) #R(2+1)D
        model = CNNClassifayer()
        model.load_state_dict(torch.load("/home/dsi/ohadico97/homework/checkpoint_3DCNN_0.001_16.pth"))
        model.to(device)
        _,_,test_data_l = prep_data()
        test(model,test_data_l,device)
