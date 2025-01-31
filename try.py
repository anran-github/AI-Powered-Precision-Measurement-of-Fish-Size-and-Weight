import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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

    # Plot the data
    plt.scatter(true_labels_sorted, predicted_data_sorted, color='blue', label='Data points')
    plt.plot(true_labels_sorted, predicted_fit, color='red', label=f'Fit line (R² = {r2:.2f})')

    # Add labels, legend, and title
    plt.xlabel('True Labels (Sorted)')
    plt.ylabel('Predicted Data')
    plt.title('True vs Predicted with Fit Line')
    plt.legend()
    plt.grid(True)
    plt.show()
