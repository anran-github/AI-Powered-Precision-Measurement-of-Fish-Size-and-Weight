import torch
import torch.nn as nn


def skip_block(input_size,out_size):
    return nn.Sequential(
        nn.Linear(input_size, out_size),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(out_size)
        
    )

class WeightNet(nn.Module):
    def __init__(self, input_size=3, hidden_size=64,output_size=1):
        super(WeightNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.batchnorm = nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2+input_size*2, hidden_size*3)
        self.fc4 = nn.Linear(hidden_size*3, hidden_size*4)
        self.fc5 = nn.Linear(hidden_size*4+hidden_size*2+input_size*2, hidden_size*4)
        self.fc6 = nn.Linear(hidden_size*4, output_size)
        self.dropout = nn.Dropout(0.5)
        self.skip_connection1 = skip_block(input_size,input_size*2)
        self.skip_connection2 = skip_block(hidden_size*2+input_size*2,hidden_size*2+input_size*2)
        self.relu = nn.ReLU()


    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        # x = torch.relu(self.batchnorm(self.fc1(x)))
        x_tmp1 = self.skip_connection1(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        x_tmp2 = self.skip_connection2(torch.concatenate((x,x_tmp1),dim=1))
    
        x = self.relu(self.fc3(torch.concatenate((x,x_tmp1),dim=1)))
        x = self.dropout(x)

        x = self.relu(self.fc4(x))
        x =  self.relu(self.fc5(torch.concatenate((x,x_tmp2),dim=1)))
        x = self.dropout(x)

        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        
        return x.view(x.size(0), -1)  # Reshape back to (2, n)


class WeightNet0(nn.Module):
    def __init__(self, input_size=3, hidden_size1=8, hidden_size2=16, output_size=1):
        super(WeightNet0, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.batchnorm = nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2*2)
        self.fc2_1 = nn.Linear(hidden_size2*2, hidden_size2*4)
        self.fc2_1_1 = nn.Linear(hidden_size2*4, hidden_size2*4)
        self.fc2_1_2 = nn.Linear(hidden_size2*4, hidden_size2*2)
        self.fc2_2 = nn.Linear(hidden_size2*2, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        # x = torch.relu(self.batchnorm(self.fc1(x)))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2_1(x))
        x = torch.relu(self.fc2_1_1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2_1_2(x)) 
        x = torch.relu(self.fc2_2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        
        return x.view(x.size(0), -1)  # Reshape back to (2, n)


'''
# Define input and output sizes
input_size = 3
output_size = 1

# Define batch size (n)
batch_size = 5

# Create the fully connected layer model
net = WeightNet()
print(net)
# Generate some random input data for demonstration
input_data = torch.randn(batch_size,3)

# Apply the fully connected layer
output = net(input_data)

# Print the output shape
print(output.shape)

'''