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
from glob import glob
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
parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size',type=int, default=1024, help='batch size')

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



criterion = nn.MSELoss()




init_test_loss = []

def test(model, test_loader,epoch):
    test_loss = 0.0
    model.eval()
    loss_set = []
    with torch.no_grad():
        loop = tqdm(test_loader)
        for img_name, inputs, targets in loop:
            outputs = model(inputs.to(device))
            loss = criterion(outputs.squeeze(1), targets.to(device))

            outputs = outputs.cpu().detach().numpy()
            targets = targets.numpy()
            # print(50*'=')
            # # print(data_test.min_vals[-2]+(data_test.max_vals[-2]-data_test.min_vals[-2])*outputs[:20,:].T)
            # # print(data_test.min_vals[-2]+(data_test.max_vals[-2]-data_test.min_vals[-2])*targets[:20])
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
    # test_loss /= len(test_loader.dataset)
    # print(f'Test Loss: {test_loss:.4f}')


    # Predict using the loaded best model
    y_pred = outputs.copy()

    sort_mask = np.argsort(targets)
    targets = targets[sort_mask]
    y_pred = y_pred[sort_mask]
    img_name = np.array(img_name)
    image_sorted = img_name[sort_mask]

    # Calculate error
    error = np.abs(targets - y_pred[:,0])
    print(f'\nWeight Error:\nAverage: {np.mean(error)} g,',
            f'\nMax : {np.max(error)} g,',
            f'\nMin : {np.min(error)} g,',
            f'\nMediam : {np.median(error)} g')

    print(f'max error image: {image_sorted[np.argmax(error)]}')
    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the first dataset (predicted weights)
    ax1.bar(range(y_pred.shape[0]), y_pred[:,0], width=0.8, align='center', label='Predicted Weights', alpha=0.5)

    # Plot the second dataset (true weights), shifted to the right
    ax1.bar(np.arange(y_pred.shape[0]), targets, width=0.8, align='center', label='True Weights', alpha=0.5)

    # Set labels and title for the first y-axis
    ax1.set_xlabel('Images')
    ax1.set_ylabel('Weights [g]')
    ax1.set_title('Block Diagram with Two Datasets and Error')
    ax1.legend(loc='upper left')

    # Create a second y-axis for the error line plot
    ax2 = ax1.twinx()
    ax2.plot(range(error.shape[0]), error, color='red', marker='o', label='Error')
    ax2.set_ylabel('Error [g]')

    # Add a legend for the second y-axis
    ax2.legend(loc='upper right')

    plt.savefig('NN_result.png')

    # Show the plot
    plt.show()


    # save model
    # Save the model
    # if len(init_test_loss)==0:


# weights_path = glob('fish_saved_weights/model_epoch*.pth')
weights_path = ['fish_saved_weights/model_epoch80_0.15009590983390808.pth']

'''
Currently good weights:
fish_saved_weights/model_epoch55_1.1315820217132568.pth

'''
for pre_trained_path in weights_path:

    model.load_state_dict(torch.load(pre_trained_path))
    model.eval()
    print('weights: {}'.format(pre_trained_path))

    test(model, test_loader,epoch=0)


