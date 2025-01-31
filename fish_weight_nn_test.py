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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from fish_weight_dataset import WeightData
from fish_weight_model import WeightNet,WeightNet_CPR


plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "Times New Roman"

def predict_label_error_fit(true_labels,predicted_data):
    # Example data
    # true_labels = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
    # predicted_data = np.array([1.1, 2.9, 4.8, 6.7, 8.9, 2.2, 3.9, 6.1, 7.8, 9.7])

    # Sort the data by true_labels
    sorted_indices = np.argsort(true_labels)
    true_labels_sorted = true_labels[sorted_indices]
    predicted_data_sorted = predicted_data[sorted_indices]

    # Fit a linear regression model
    model = LinearRegression()
    true_labels_sorted_reshaped = true_labels_sorted.reshape(-1, 1)  # Reshape for sklearn
    model.fit(true_labels_sorted_reshaped, predicted_data_sorted)

    # Get the fit line
    predicted_fit = model.predict(true_labels_sorted_reshaped)

    # Calculate R^2
    r2 = r2_score(predicted_data_sorted, predicted_fit)
    rmse = np.sqrt(mean_squared_error(predicted_data_sorted, true_labels_sorted_reshaped))

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.scatter(true_labels_sorted, predicted_data_sorted, label='Weight Data Points')
    plt.plot(true_labels_sorted, predicted_fit, color='black',linestyle='--', linewidth=4, label=f'Fit line')
    
    # Add RMSE and R^2 values as text in the bottom-right corner
    plt.text(
        x=0.95, y=0.05,  # Coordinates relative to the axis (0.95 = 95% of x, 0.05 = 5% of y)
        s=f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nSample Size = {true_labels_sorted.shape[0]}',
        color='black',
        horizontalalignment='right', verticalalignment='bottom',
        transform=plt.gca().transAxes,  # Use axis coordinates for positioning
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    # Add labels, legend, and title
    plt.xlabel('Measured Weight [g]')
    plt.ylabel('Predicted Weight [g]')
    # plt.title('Measured Weight versus Predicted with Fit Line')
    plt.legend()
    plt.grid(linestyle = '--')
    plt.tight_layout()
    # plt.savefig('paper_image/weight_error_eval.png')
    plt.savefig('paper_image/weight_cpr_error_eval2.png')
    plt.show()


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
# model = WeightNet().to(device)
model = WeightNet_CPR().to(device)

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

def test_barplot(model, test_loader,epoch):
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

    # plt.savefig('NN_result.png')

    # Show the plot
    plt.show()


    # save model
    # Save the model
    # if len(init_test_loss)==0:

def test_paper_plot(model, test_loader,epoch):
    '''
    Plot figure for paper: RMSE, R^2 fitting
    '''
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
    # error = np.sqrt(mean_squared_error(targets, y_pred[:,0]))
    print(f'\nWeight Error:\nAverage (MAE): {np.mean(error)} g,',
            f'\nMax : {np.max(error)} g,',
            f'\nMin : {np.min(error)} g,',
            f'\nMediam : {np.median(error)} g')

    print(f'max error image: {image_sorted[np.argmax(error)]}')

    # remove max error image:
    y_pred = y_pred[:,0]
    targets = np.delete(targets,[np.argmax(error)],None)
    y_pred = np.delete(y_pred,[np.argmax(error)],None)

    predict_label_error_fit(targets,y_pred)




    # # Create figure and axes
    # fig, ax1 = plt.subplots(figsize=(10, 6))

    # # Plot the first dataset (predicted weights)
    # ax1.bar(range(y_pred.shape[0]), y_pred[:,0], width=0.8, align='center', label='Predicted Weights', alpha=0.5)

    # # Plot the second dataset (true weights), shifted to the right
    # ax1.bar(np.arange(y_pred.shape[0]), targets, width=0.8, align='center', label='True Weights', alpha=0.5)

    # # Set labels and title for the first y-axis
    # ax1.set_xlabel('Images')
    # ax1.set_ylabel('Weights [g]')
    # ax1.set_title('Block Diagram with Two Datasets and Error')
    # ax1.legend(loc='upper left')

    # # Create a second y-axis for the error line plot
    # ax2 = ax1.twinx()
    # ax2.plot(range(error.shape[0]), error, color='red', marker='o', label='Error')
    # ax2.set_ylabel('Error [g]')

    # # Add a legend for the second y-axis
    # ax2.legend(loc='upper right')

    # plt.savefig('NN_result.png')

    # # Show the plot
    # plt.show()


    # save model
    # Save the model
    # if len(init_test_loss)==0:


# weights_path = glob('fish_saved_weights/model_epoch*.pth')
# weights_path = glob('fish_cpr_saved_weights/model_epoch*.pth')
# weights_path = ['fish_saved_weights/model_epoch80_0.15009590983390808.pth']
weights_path = ['fish_cpr_saved_weights/model_epoch495_0.15008985996246338.pth']

'''
Currently good weights:
fish_saved_weights/model_epoch55_1.1315820217132568.pth

'''
for pre_trained_path in weights_path:

    model.load_state_dict(torch.load(pre_trained_path))
    model.eval()
    print('weights: {}'.format(pre_trained_path))

    # test_barplot(model, test_loader,epoch=0)
    test_paper_plot(model, test_loader,epoch=0)


