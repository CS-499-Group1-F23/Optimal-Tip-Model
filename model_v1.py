# pip install pandas scikit-learn seaborn matplotlib

import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
def load_data():
    order_data = pd.read_csv('Data/order-initial-dataset.csv')
    store_data = pd.read_csv('Data/store-initial-dataset.csv')
    return order_data, store_data

# Preprocess the data
#   Preprocesses the order and store data for machine learning.

#   Args:
#     order_data: A Pandas DataFrame containing the order data.
#     store_data: A Pandas DataFrame containing the store data.

#   Returns:
#     A Pandas DataFrame containing the preprocessed data.
def preprocess_data(order_data, store_data):
    order_data = order_data[order_data['Destination_type'] == 'Delivery'] #Drop non Delivery orders
    order_data = order_data[~order_data['Source_actor'].isin(['ubereats', 'doordash', 'grubhub'])] # Drop 3rd party aggregetors
    merged_data = pd.merge(order_data, store_data, on='Store_dma_id', how='inner') # Merge two datapoints using Store_dma_id as primary_key
    return merged_data

def data_loader(merged_data, test_size=0.2, percentage_zero_dollar_tip=0.1): 
#    Loads and prepares data for training and testing a machine learning model.

#   Args:
#     merged_data: A Pandas DataFrame containing the merged dataset.
#     test_size: The percentage of data to use for testing post training.
#     percentage_zero_dollar_tip: The percentage of the dataset that have as tips == 0.

#   Returns:
#     A tuple of (train_data, test_data), where train_data is a Pandas DataFrame
#     containing the training data and test_data is a Pandas DataFrame containing
#     the testing data.

    if percentage_zero_dollar_tip > merged_data['Tip_USD'].value_counts(normalize=True).get(0, 0):
        raise ValueError("Invalid percentage. Not enough data points with zero tips.")
    X = merged_data[['Store_dma_id', 'Subtotal_amount_USD']]
    y = merged_data['Tip_USD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    
#   Trains and evaluates a machine learning model.

#   Args:
#     X_train: A Pandas DataFrame containing the training features.
#     X_test: A Pandas DataFrame containing the testing features.
#     y_train: A Pandas Series containing the training labels.
#     y_test: A Pandas Series containing the testing labels.

#   Returns:
#     A tuple of (model, accuracy), where model is a trained machine learning model
#     and accuracy is the model's accuracy on the testing set.
  
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    accuracy = 1 - (mse / y_test.var())
    return model, accuracy

# Visualization
def visualize_correlation(merged_data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Rack_time', y='Tip_USD', data=merged_data)
    plt.xlabel('Rack Time')
    plt.ylabel('Tip (USD)')
    plt.title('Correlation between Rack Time and Tip')
    plt.show()


def main(visualize=False, save_artifacts=False):
    order_data, store_data = load_data()
    merged_data = preprocess_data(order_data, store_data)
    X_train, X_test, y_train, y_test = data_loader(merged_data, test_size=0.2, percentage_zero_dollar_tip=0.1)
    model, accuracy = train_model(X_train, X_test, y_train, y_test)
    print(f'Model Accuracy: {accuracy:.2f}')

    if save_artifacts:
        # Save merged_data to a CSV file with the specified naming convention
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'merged_data_{current_datetime}.csv'
        merged_data.to_csv(file_name, index=False)
        print(f"Merged data saved as '{file_name}'.")

    if visualize:
        visualize_correlation(merged_data)
        # Additional visualization for linear regression can be added here.

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process data and perform analysis.')
    parser.add_argument('-V', '--visualize', action='store_true', help='Visualize correlation graphs.')
    parser.add_argument('-A', '--artifacts', action='store_true', help='Save merged data to CSV.')
    args = parser.parse_args()

    main(visualize=args.visualize, save_artifacts=args.artifacts)
