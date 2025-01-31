'''
Draw three graphs:

1. Length Dataset distribution

2. Weight Dataset distribution

3. Length Error Estimation

'''


import torch
import numpy as np
from  torch.utils.data import Dataset, random_split
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error



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
        s=f'R² = {r2:.2f}\nRMSE = {rmse:.2f}\nSample Size = {true_labels_sorted.shape[0]}',
        color='black',
        horizontalalignment='right', verticalalignment='bottom',
        transform=plt.gca().transAxes,  # Use axis coordinates for positioning
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    # Add labels, legend, and title
    plt.xlabel('Measured Length [cm]')
    plt.ylabel('Predicted Length [cm]')
    # plt.title('Measured Weight versus Predicted with Fit Line')
    plt.legend()
    plt.grid(linestyle = '--')
    plt.tight_layout()
    plt.savefig('paper_image/length__noopt_error_eval.png')
    plt.show()


class WeightData(Dataset):
    '''
    Generate data for weight prediction.

        input path :  File path saves bbox width, height, areas. Generate 
                      from 'fish_widthheight_area_dataset_generator.py'

        label path :  Excel file which saved true weight and width data

        mode       :  Open specific training/testing file

    
    '''
    def __init__(self,input_path, label_path,mode='train'):
        '''
        Generate weight dataset for weight prediction.
        '''

        self.mode = mode

        with open(input_path,'r') as f:
            self.input_data = json.load(f)

        self.dataset_path_train = 'weight_noopt_dataset_train.json'
        self.dataset_path_test = 'weight_noopt_dataset_test.json'

        # add extended data sheet here if you have:
        self.dataset_dict = {'date':['12-11-24','12-18-24','12-30-24','Tk 4 - varied data','Tk 5 - varied data'],
                        'sheet':["D2 Growth Study 12-11","D3 Growth Study 12-18","D4 Growth Study 12-30","Tank4","Tank5"],
                        'pixel_size':[0.0038845388570005477,  
                                      0.0039034090258904977, 
                                      0.003898025573224454, 
                                      0.0038995183904531544, 
                                      0.0039154904929156725]}  
                                    # 0.004005

        self.length_summary_data = {}
        
        if not os.path.exists(self.dataset_path_train):
            print('\n Generating the dataset...')
            
            self.valid_dataset = []
            for i in range(len(self.dataset_dict['date'])):  
                date_data = self.dataset_dict['date'][i]
                pixel_size = self.dataset_dict['pixel_size'][i]
                self.label_data = pd.read_excel(label_path, sheet_name=self.dataset_dict['sheet'][i])
                self.valid_num = 0

                # only consider data with valid weight info:
                while pd.notna(self.label_data.iloc[self.valid_num,1]):

                    image_id = self.label_data.iloc[self.valid_num,0]


                    # generate training/testing data as following format:
                    #       RUNNING FOR THE FIRST TIME
                    #    SAVED IN: 'weight_dataset.json'
                    # ---------------------------------------------------
                    #   image_path, w, h, area, true_weight, true_length 
                    # ---------------------------------------------------
                    image_name = '/' + str(image_id) + '.JPG'
                    true_weightdata = self.label_data.iloc[self.valid_num,1]

                    combined_data = [x for x in self.input_data if image_name in x[0] and date_data in x[0]]
                    
                    if len(combined_data) == 1:

                        # pixel --> cm:
                        combined_data[0][1] *= pixel_size
                        combined_data[0][2] *= pixel_size
                        combined_data[0][3] *= pixel_size**2


                        combined_data[0].append(true_weightdata)

                        # save length info
                        if pd.notna(self.label_data.iloc[self.valid_num,3]):
                            combined_data[0].append(self.label_data.iloc[self.valid_num,3])
                        else:
                            combined_data[0].append(0)

                        self.valid_dataset.append(combined_data[0])
                    else:
                        print('No exact file/Multiple files are found for a single image name!')
                        print(image_id,date_data,self.dataset_dict['sheet'][i])


                    self.valid_num += 1

            # save valid dataset for training/testing:
            generator1 = torch.Generator().manual_seed(42)
            train_list, test_list = random_split(self.valid_dataset,[0.9,0.1],generator=generator1)
            
            with open(self.dataset_path_train,'w') as f:
                self.valid_dataset_train = [self.valid_dataset[x] for x in train_list.indices]
                json.dump(self.valid_dataset_train,f)

            with open(self.dataset_path_test,'w') as f:
                self.valid_dataset_test = [self.valid_dataset[x] for x in test_list.indices]
                json.dump(self.valid_dataset_test,f)

        else:
            print('\n Opening existing dataset ... ')

            if self.mode == 'train':
                with open(self.dataset_path_train,'r') as fp:
                    self.valid_dataset_train = json.load(fp)
            else:
                with open(self.dataset_path_test,'r') as fp:
                    self.valid_dataset_test = json.load(fp)

        
        # get max and min for normilization:
        with open(self.dataset_path_train,'r') as fp:
            self.valid_dataset_train = json.load(fp)

        with open(self.dataset_path_test,'r') as fp:
            self.valid_dataset_test = json.load(fp)

        self.dataset_total = np.concatenate((self.valid_dataset_train,self.valid_dataset_test),axis=0)
        data_tmp = np.array(self.dataset_total)
        self.max_vals = np.max(np.array(data_tmp[:,1:],np.float32),axis=0)
        self.min_vals = np.min(np.array(data_tmp[:,1:],np.float32),axis=0)


        print('\n Dataset is ready ... ')
        print(f'Valid Training Set Number: {len(self.valid_dataset_train)}')
        print(f'Valid Testing Set Number: {len(self.valid_dataset_test)}')
        print(50*'=')            
         
    
    def __getitem__(self, index):
        # TO DO      :  in real case, we should open an image and do segmentation.
        if self.mode == 'train':
            data_idx = self.valid_dataset_train[index]
        else:
            data_idx = self.valid_dataset_test[index]
        pixel_size = 0

        for i,date in enumerate(self.dataset_dict['date']):
            if date in data_idx[0]:        
                pixel_size = self.dataset_dict['pixel_size'][i]
        if pixel_size == 0:
            print('Pixel Size Cannot be ZERO!')

        # input data : w, h, area
        # label      : true_weight

        input_data = [data_idx[1],data_idx[2],data_idx[3]]
        
        # norm data
        # input_data = (np.array(input_data)-self.min_vals[:3])/(self.max_vals[:3]-self.min_vals[:3])
        # label = (data_idx[-2]-self.min_vals[-2])/(self.max_vals[-2]-self.min_vals[-2])

        # keep real data:
        label = data_idx[-2]
        
        # print('====================================')
        # print(data_idx[0])
        # print(input_data,label)
        input_img_name = data_idx[0]
        input_data = torch.tensor(input_data,dtype=torch.float32)
        label = torch.tensor(label,dtype=torch.float32)
        return input_img_name, input_data, label
    
    def __len__(self,):
        if self.mode == 'train':
            return len(self.valid_dataset_train)
        else:
            return len(self.valid_dataset_test)
        
    def length_summary(self,):
        # evaluate detected length vs real length

        with open(self.dataset_path_train,'r') as fp:
            valid_dataset_train = json.load(fp)

        with open(self.dataset_path_test,'r') as fp:
            valid_dataset_test = json.load(fp)

        # ---------------------------------------------------
        #   image_path, w, h, area, true_weight, true_length 
        # ---------------------------------------------------
        
        total_dataset = np.concatenate((valid_dataset_train,valid_dataset_test),axis=0)
        estimated_length = np.array(total_dataset[:,1:3],np.float32)
        estimated_length = np.max(estimated_length,axis=1)
        real_length      = np.array(total_dataset[:,5],np.float32)

        valid_est_length = estimated_length[real_length > 0]
        valid_image = total_dataset[real_length > 0]
        valida_real_length = real_length[real_length > 0]

        error = np.abs(valida_real_length-valid_est_length)
        
        print(f'Valid Length Data Number: {valid_est_length.shape[0]} \n')
        print(f'\nLength Error:\nAverage: {np.mean(error)} cm,',
              f'\nMax Error: {np.max(error)} cm,',
              f'\nMin Error: {np.min(error)} cm,',
              f'\nMediam: {np.median(error)} cm')
        
        img_name = valid_image[np.argsort(error)[::-1]]
        topk = 3
        print(f'Error for each image:\n {error[np.argsort(error)[::-1][:topk]]}')
        print(f'Problem Image:\n {img_name[:topk,0]}')


        # show error with fit line: APPEAR IN PAPER!
        valida_real_length = np.delete(valida_real_length,[np.argmax(error)],None)
        valid_est_length = np.delete(valid_est_length,[np.argmax(error)],None)
        predict_label_error_fit(valida_real_length,valid_est_length)

        # shown error with bar graph
        '''
        # Predict using the loaded best model
        y_pred = valid_est_length.copy()
        targets = valida_real_length.copy()


        # Create figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the first dataset (predicted weights)
        ax1.bar(range(y_pred.shape[0]), y_pred, width=0.8, align='center', label='Predicted Length', alpha=0.5)

        # Plot the second dataset (true weights), shifted to the right
        ax1.bar(np.arange(y_pred.shape[0]), targets, width=0.8, align='center', label='True Length', alpha=0.5)

        # Set labels and title for the first y-axis
        ax1.set_xlabel('Images')
        ax1.set_ylabel('Length [cm]')
        ax1.set_title('Length Prediction Result')
        ax1.legend(loc='upper left')

        # Create a second y-axis for the error line plot
        ax2 = ax1.twinx()
        ax2.plot(range(error.shape[0]), error, color='red', marker='o', label='Error')
        ax2.set_ylabel('Error [cm]')

        # Add a legend for the second y-axis
        ax2.legend(loc='upper right')
        
        plt.savefig('Length_result.png')

        # Show the plot
        plt.show()
        plt.close()
        '''


        # Analyze the distribution
        mean = np.mean(valida_real_length)
        std_dev = np.std(valida_real_length)

        # Plot the histogram of the data
        plt.hist(valida_real_length, bins=30, density=True, alpha=0.6, color='orange', label='Length Data Histogram')

        # Overlay a normal distribution for comparison
        x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
        normal_dist = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)

        plt.plot(x, normal_dist, color='red', label='Normal Distribution Fit')

        # Add titles and labels
        plt.title('Length Data Distribution')
        plt.xlabel('Value [cm]')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(linestyle='--')
        plt.tight_layout()
        plt.savefig('paper_image/length_data_sum.png')

        # Show the plot
        plt.show()


    def weight_summary(self,):
        # evaluate weights distribution

        with open(self.dataset_path_train,'r') as fp:
            valid_dataset_train = json.load(fp)

        with open(self.dataset_path_test,'r') as fp:
            valid_dataset_test = json.load(fp)
        
        total_dataset = np.concatenate((valid_dataset_train,valid_dataset_test),axis=0)
        true_weight = np.array(total_dataset[:,4],np.float32).reshape(-1,1)
        
        print(f'Valid Weight Data Number: {true_weight.shape[0]} \n')

        print(f'\nWeight:\nAverage: {np.mean(true_weight)} g,',
              f'\nMax Weight: {np.max(true_weight)} g,',
              f'\nMin Weight: {np.min(true_weight)} g,',
              f'\nMediam: {np.median(true_weight)} g')

        # Sample data (replace this with your dataset)
        # data = np.random.normal(loc=0, scale=1, size=1000)  # Example: normally distributed data

        # Analyze the distribution
        mean = np.mean(true_weight)
        std_dev = np.std(true_weight)

        # Plot the histogram of the data
        plt.hist(true_weight, bins=30, density=True, alpha=0.6, color='blue', label='Weight Data Histogram')

        # Overlay a normal distribution for comparison
        x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
        normal_dist = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)

        plt.plot(x, normal_dist, color='red', label='Normal Distribution Fit')

        # Add titles and labels
        plt.title('Weight Data Distribution')
        plt.xlabel('Value [g]')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(linestyle='--')
        plt.tight_layout()
        plt.savefig('paper_image/weight_data_sum.png')

        # Show the plot
        plt.show()


# '''
from torch.utils.data import DataLoader
data = WeightData(input_path='bbox_area_dataset_no_bbox_optimization.json',label_path='/media/anranli/DATA/data/fish/Growth Study Data 12-2024.xlsx',mode='train')
data.length_summary()
# data.weight_summary()

# '''