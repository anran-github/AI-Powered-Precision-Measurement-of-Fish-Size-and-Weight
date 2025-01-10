import matplotlib.pyplot as plt
import numpy as np

# Example data (replace these with your actual y_pred and y_test)
y_pred = np.array([3.5, 4.2, 5.1, 4.8, 3.9])
y_test = np.array([3.0, 4.5, 5.0, 4.7, 4.0])

# Calculate error
error = y_test - y_pred

# Create figure and axes
fig, ax1 = plt.subplots()

# Plot the first dataset (predicted weights)
ax1.bar(range(y_pred.shape[0]), y_pred, width=0.4, align='center', label='Predicted Weights', alpha=0.5)

# Plot the second dataset (true weights), shifted to the right
ax1.bar(np.arange(y_pred.shape[0]) + 0.4, y_test, width=0.4, align='center', label='True Weights', alpha=0.5)

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

# Show the plot
plt.show()
