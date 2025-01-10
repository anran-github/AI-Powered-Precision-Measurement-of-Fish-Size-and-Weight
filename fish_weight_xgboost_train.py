import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# plot
import matplotlib.pyplot as plt


from fish_weight_dataset import WeightData

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_path',type=str,default='bbox_area_dataset.json', help='file with input bbox info and areas')
parser.add_argument('--label_excel',type=str,default='/media/anranli/DATA/data/fish/Growth Study Data 12-2024.xlsx', help='Ground Truth Weight Label')
parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size',type=int, default=1024, help='batch size')
parser.add_argument('--total_epoch',type=int, default=1000, help='batch size')
parser.add_argument('--pre_trained', type=str, default='',help='input your pretrained weight path if you want')

args = parser.parse_args()

print(args)




# import dataset
data_train = WeightData(input_path=args.input_path,label_path=args.label_excel,mode='train')
data_test = WeightData(input_path=args.input_path,label_path=args.label_excel,mode='test')


train_loader = DataLoader(data_train,batch_size=args.batch_size,shuffle=True)
test_loader = DataLoader(data_test,batch_size=args.batch_size,shuffle=False)

for train_data, test_data in zip(train_loader,test_loader):
    train_img_name, X_train ,y_train   = train_data
    test_img_name,  X_test  ,y_test    = test_data

print(test_img_name)
# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train.numpy(), label=y_train.numpy())
dtest = xgb.DMatrix(X_test.numpy(), label=y_test.numpy())
# Set XGBoost parameters for regression with GPU support
params = {
    "objective": "reg:squarederror",  # Regression objective
    "max_depth": 3,
    "eta": 0.1,
    "alpha":10,
    # "gamma":1,
    "tree_method": "gpu_hist",  # Use GPU for training
    "eval_metric": "rmse"  # Root Mean Squared Error
}

# Define a custom callback to save the best model
class SaveBestModelCallback(xgb.callback.TrainingCallback):
    def __init__(self):
        self.best_score = float("inf")
        self.best_model_path = 'weight_xgboost/best_xgboost_model.bin'

    def after_iteration(self, model, epoch, evals_log):
        # Check for the "test" set in evaluation logs
        if "test" in evals_log:
            current_score = evals_log["test"]["rmse"][-1]
            if current_score < self.best_score:
                self.best_score = current_score
                model.save_model(self.best_model_path)
                print(f"New best model saved with test RMSE: {self.best_score:.4f}")
        return False  # Returning False continues training

# Instantiate the callback
save_best_model_callback = SaveBestModelCallback()



if len(args.pre_trained) == 0:
    print('Training MODEL ...')

    # Train the model with the callback
    num_round = 100
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_round,
        evals=[(dtest, "test")],
        callbacks=[save_best_model_callback],
    )



print('Load pre-trained xgboost model ...')
# Load the saved best model
loaded_best_model = xgb.Booster()
loaded_best_model.load_model(save_best_model_callback.best_model_path)
print(f"Loaded best model with RMSE: {save_best_model_callback.best_score:.4f}")

# Predict using the loaded best model
y_pred = loaded_best_model.predict(dtest)

# Evaluate performance using RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE of the best model: {rmse:.4f}")


sort_mask = np.argsort(y_test)
y_test = y_test[sort_mask].numpy()
y_pred = y_pred[sort_mask]
# Calculate error
error = np.abs(y_test - y_pred)

print(f'\nWeight Error:\nAverage: {np.mean(error)} g,',
        f'\nMax : {np.max(error)} g,',
        f'\nMin : {np.min(error)} g,',
        f'\nMedian : {np.median(error)} g')


# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the first dataset (predicted weights)
ax1.bar(range(y_pred.shape[0]), y_pred, width=0.8, align='center', label='Predicted Weights', alpha=0.5)

# Plot the second dataset (true weights), shifted to the right
ax1.bar(np.arange(y_pred.shape[0]), y_test, width=0.8, align='center', label='True Weights', alpha=0.5)

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
plt.savefig('XGBoost_result.png')
# Show the plot
plt.show()
