import json
import pandas as pd
# from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse

from fish_weight_dataset import WeightData
from fish_weight_model import WeightNet


# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"\n Using device: {device}")


# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_path',type=str,default='bbox_area_dataset.json', help='file with input bbox info and areas')
parser.add_argument('--label_excel',type=str,default='/media/anranli/DATA/data/fish/Growth Study Data 12-2024.xlsx', help='Ground Truth Weight Label')
parser.add_argument('--lr',type=float, default=0.00002, help='learning rate')
parser.add_argument('--batch_size',type=int, default=1024, help='batch size')
parser.add_argument('--total_epoch',type=int, default=500, help='batch size')
parser.add_argument('--pre_trained', type=str, default='fish_saved_weights/model_epoch80_0.15009590983390808.pth',help='input your pretrained weight path if you want')

args = parser.parse_args()

print(args)




# import network
model = WeightNet().to(device)

# import dataset
data_train = WeightData(input_path=args.input_path,label_path=args.label_excel,mode='train')
data_test = WeightData(input_path=args.input_path,label_path=args.label_excel,mode='test')


train_loader = DataLoader(data_train,batch_size=args.batch_size,shuffle=True)
test_loader = DataLoader(data_test,batch_size=args.batch_size,shuffle=False)


save_path = f'fish_saved_weights'
if not os.path.exists(save_path):
    os.mkdir(save_path)


if len(args.pre_trained):
    model.load_state_dict(torch.load(args.pre_trained,weights_only=True))
    model.eval()
    print('----------added previous weights: {}------------'.format(args.pre_trained))

criterion = nn.MSELoss(reduction='none')
criterion_test = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epoch//5, eta_min=args.lr*0.1)


init_test_loss = []

def test(model, test_loader,epoch):
    test_loss = 0.0
    model.eval()
    loss_set = []
    with torch.no_grad():
        loop = tqdm(test_loader)
        for img_name, inputs, targets in loop:
            outputs = model(inputs.to(device))
            loss = criterion_test(outputs, targets.to(device))
            # print(50*'=')
            # print(outputs[:20,:].T)
            # print(targets[:20])
            # print(loss)
            if not 'cuda' in device.type:
                loss_set.append(loss.item())
                test_loss += loss.item() * inputs.size(0)
            else:
                loss_set.append(loss.cpu().item())
                test_loss += loss.cpu().item() * inputs.size(0)

            loop.set_postfix(loss=f"{loss_set[-1]:.4f}", refresh=True)
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')


    outputs = outputs.cpu().detach().numpy()
    targets = targets.numpy()

    # Predict using the loaded best model
    y_pred = outputs.copy()

    sort_mask = np.argsort(targets)
    targets = targets[sort_mask]
    y_pred = y_pred[sort_mask]

    # Calculate error
    error = np.abs(targets - y_pred[:,0])
    print(f'\nWeight Error:\nAverage: {np.mean(error)} g,',
            f'\nMax : {np.max(error)} g,',
            f'\nMin : {np.min(error)} g,',
            f'\nMediam : {np.median(error)} g')



    # save model
    # Save the model
    # if len(init_test_loss)==0:
    init_test_loss.append(np.mean(error))
    if len(init_test_loss) >= 2:
        if init_test_loss[-1] == min(init_test_loss) and init_test_loss[-1] < init_test_loss[-2] and epoch != 0:
            print('Model Saved!')
            torch.save(model.state_dict(), os.path.join(save_path,'model_epoch{}_{}.pth'.format(epoch,init_test_loss[-1])))




# test(model, test_loader,epoch=0)



losses = []
loss_avg = []
model.train()
for epoch in range(args.total_epoch):
    for img_name, inputs, targets in train_loader:

        # get frequency of label, weight rare data:
        # Compute label frequencies
        
        unique_labels, counts = torch.unique(targets, return_counts=True)
        label_weights = 1.0 / counts.float()  # Inverse frequency for weights
        weights = torch.zeros_like(targets, dtype=torch.float)

        # Assign weights based on label frequency
        for label, weight in zip(unique_labels, label_weights):
            weights[targets == label] = weight
        weights = weights.to(device)


        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        mse_loss = criterion(outputs.squeeze(1), targets.to(device))

        # uncomment to introduce weights:
        loss = (weights * mse_loss).sum() 

        # loss = (mse_loss).sum() 



        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100)

        optimizer.step()

        # Update the learning rate
        scheduler.step() 

        loss_avg.append(loss.item())  # Store the loss value for plotting

    print(f'Epoch [{epoch+1}/{args.total_epoch}], Loss: {loss.item():.4f}')
    losses.append(np.average(loss_avg))

    if epoch >= 5 and epoch%5==0:
        test(model, test_loader,epoch)

# Plot the loss dynamically
plt.clf()  # Clear previous plot
plt.plot(losses, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
# plt.pause(0.05)  # Pause for a short time to update the plot
plt.savefig(os.path.join(save_path,'training_loss_{}.png'.format(args.total_epoch)))
plt.plot()




