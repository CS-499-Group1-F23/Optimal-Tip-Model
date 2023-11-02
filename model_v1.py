# pip install pandas scikit-learn seaborn matplotlib
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import logging


# Set up logging configuration
logging.basicConfig(filename='Data/log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# load the dataset
def load_data():
    #order_data = pd.read_csv('Data/order-initial-dataset.csv')
    order_data = pd.read_csv('Data/order_data_10-30.csv')
    store_data = pd.read_csv('Data/store_data_10-16.csv')
    return order_data, store_data


def preprocess_data(order_data, store_data, tip_percentage):
    order_data = order_data[order_data['Destination_type'] == 'Delivery'] #Drop non Delivery orders
    order_data = order_data[~order_data['Source_actor'].isin(['ubereats', 'doordash', 'grubhub'])] # Drop 3rd party aggregetors
    merged_data = pd.merge(order_data, store_data, on='store_number', how='inner') # Merge two datapoints using Store_dma_id as primary_key
    merged_data['total_amount_USD'] = merged_data['total_tax_USD'] + merged_data['subtotal_amount_USD'] # Sum post tax
    merged_data['good_tip'] = merged_data.apply(lambda row: 'TRUE' if row['Tip_USD'] > tip_percentage * row['total_amount_USD'] else ('ZERO' if row['Tip_USD'] == 0 else 'FALSE'), axis=1) # Get good tip 
    
    total_rows = len(merged_data)
    good_tip_percentage = len(merged_data[merged_data['good_tip'] == 'TRUE']) / total_rows * 100
    bad_tip_percentage = len(merged_data[merged_data['good_tip'] == 'FALSE']) / total_rows * 100
    zero_tip_percentage = len(merged_data[merged_data['good_tip'] == 'ZERO']) / total_rows * 100
    print(f"Percentage of good tip data: {good_tip_percentage:.2f}%")
    print(f"Percentage of bad tip data: {bad_tip_percentage:.2f}%")
    print(f"Percentage of zero tip data: {zero_tip_percentage:.2f}%")
    print("data size", merged_data.size)
    logging.info(f"Percentage of good tip data: {good_tip_percentage:.2f}%")
    logging.info(f"Percentage of bad tip data: {bad_tip_percentage:.2f}%")
    logging.info(f"Percentage of zero tip data: {zero_tip_percentage:.2f}%")
    logging.info("data size %d", merged_data.size)
    return merged_data

import numpy as np

def data_loader(merged_data, test_size, percentage_zero_dollar_tip, percentage_bad_tip=None, percentage_good_tip=None):
    logging.info(f'data_loader parameters - test_size: {test_size}, '
                 f'percentage_zero_dollar_tip: {percentage_zero_dollar_tip}, '
                 f'percentage_bad_tip: {percentage_bad_tip}, '
                 f'percentage_good_tip: {percentage_good_tip}')
    if percentage_zero_dollar_tip > merged_data['Tip_USD'].value_counts(normalize=True).get(0, 0):
        raise ValueError("Invalid percentage. Not enough data points with zero tips.")
    
    # Filter data based on percentage of zero tips
    zero_tip_mask = merged_data['Tip_USD'] == 0
    zero_tip_indices = zero_tip_mask.index.to_numpy()
    np.random.shuffle(zero_tip_indices)
    num_zero_tips_to_keep = int(percentage_zero_dollar_tip * len(zero_tip_indices))
    zero_tip_indices_to_keep = zero_tip_indices[:num_zero_tips_to_keep]
    merged_data = merged_data.loc[zero_tip_indices_to_keep]

    # Filter data based on percentage of bad tips
    if percentage_bad_tip:
        bad_tip_mask = merged_data['good_tip'] == 'FALSE'
        bad_tip_indices = bad_tip_mask.index.to_numpy()
        np.random.shuffle(bad_tip_indices)
        num_bad_tips_to_keep = int(percentage_bad_tip * len(bad_tip_indices))
        bad_tip_indices_to_keep = bad_tip_indices[:num_bad_tips_to_keep]
        merged_data = merged_data.loc[bad_tip_indices_to_keep]

    # Filter data based on percentage of good tips
    if percentage_good_tip:
        good_tip_mask = merged_data['good_tip'] == 'TRUE'
        good_tip_indices = good_tip_mask.index.to_numpy()
        np.random.shuffle(good_tip_indices)
        num_good_tips_to_keep = int(percentage_good_tip * len(good_tip_indices))
        good_tip_indices_to_keep = good_tip_indices[:num_good_tips_to_keep]
        merged_data = merged_data.loc[good_tip_indices_to_keep]

    X = merged_data[['store_number', 'subtotal_amount_USD']]
    y = merged_data['Tip_USD']
    print(len(X), len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test



def train_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    accuracy = 1 - (mse / y_test.var())
    return model, accuracy


def visualize_correlation(merged_data):
    merged_data_copy = merged_data.copy()

    # calculate the average rack time when the tip is zero
    avg_rack_time_zero_tip = merged_data[merged_data['Tip_USD'] == 0]['Rack_time'].mean()
    print(f"Average Rack Time when Tip is Zero: {avg_rack_time_zero_tip:.2f} minutes")

    # calculate the average rack time when the tip is more than zero
    avg_rack_time_non_zero_tip = merged_data[merged_data['Tip_USD'] > 0]['Rack_time'].mean()
    print(f"Average Rack Time when Tip is More than Zero: {avg_rack_time_non_zero_tip:.2f} minutes")

    # scatterplot before removing zero-dollar tips
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='Rack_time', y='Tip_USD', data=merged_data)
    plt.xlabel('Rack Time')
    plt.ylabel('Tip (USD)')
    plt.title('Correlation between Rack Time and Tip (Scatterplot) - Before Removal')
    plt.xlim(1, 50)
    plt.ylim(0, 50)

    # Line chart before removing zero-dollar tips
    plt.subplot(2, 2, 2)
    sns.lineplot(x='Rack_time', y='Tip_USD', data=merged_data)
    plt.xlabel('Rack Time')
    plt.ylabel('Tip (USD)')
    plt.title('Correlation between Rack Time and Tip (Line Chart) - Before Removal')
    plt.xlim(1, 50)
    plt.ylim(0, 50)

    # show statistics before removal
    zero_dollar_tips_percentage_before = (len(merged_data[merged_data['Tip_USD'] == 0]) / len(merged_data)) * 100
    correlation_coefficient_before = merged_data['Rack_time'].corr(merged_data['Tip_USD'])
    number_of_data_points_before = len(merged_data)
    print(f"Percentage of $0 tips (Before): {zero_dollar_tips_percentage_before:.2f}%")
    print(f"Correlation coefficient (Pearson) between Rack Time and Tip (Before): {correlation_coefficient_before:.2f}")
    print(f"Number of data points (Before): {number_of_data_points_before}")

    # remove outliers (tip values over $30)
    merged_data = merged_data[merged_data['Tip_USD'] <= 30]

    # scatterplot after removing zero-dollar tips and outliers
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='Rack_time', y='Tip_USD', data=merged_data)
    plt.xlabel('Rack Time')
    plt.ylabel('Tip (USD)')
    plt.title('Correlation between Rack Time and Tip (Scatterplot) - After Removal')
    plt.xlim(1, 50)
    plt.ylim(0, 30)  # Updated ylim to accommodate the removal of outliers

    # line chart after removing zero-dollar tips and outliers
    plt.subplot(2, 2, 4)
    sns.lineplot(x='Rack_time', y='Tip_USD', data=merged_data)
    plt.xlabel('Rack Time')
    plt.ylabel('Tip (USD)')
    plt.title('Correlation between Rack Time and Tip (Line Chart) - After Removal')
    plt.xlim(1, 50)
    plt.ylim(0, 30)  # Updated ylim to accommodate the removal of outliers

    # calculate statistics after removal
    zero_dollar_tips_percentage_after = (len(merged_data[merged_data['Tip_USD'] == 0]) / len(merged_data)) * 100
    correlation_coefficient_after = merged_data['Rack_time'].corr(merged_data['Tip_USD'])
    number_of_data_points_after = len(merged_data)
    print(f"Percentage of $0 tips (After Removal): {zero_dollar_tips_percentage_after:.2f}%")
    print(f"Correlation coefficient (Pearson) between Rack Time and Tip (After Removal): {correlation_coefficient_after:.2f}")
    print(f"Number of data points (After Removal): {number_of_data_points_after}")

    plt.tight_layout()
    plt.show()


def visualize_tip_distribution(merged_data):
    # tip ranges
    tip_ranges = [0, 1, 3, 5, 8, 12, 15, 30, float('inf')]

    # Initialize counts for each tip range
    tip_counts = [0] * len(tip_ranges)

    # Calculate the counts for each tip range
    for tip in merged_data['Tip_USD']:
        for i, max_tip in enumerate(tip_ranges):
            if tip <= max_tip:
                tip_counts[i] += 1
                break

    # Calculate the percentage of tips in each range
    total_tips = len(merged_data)
    tip_percentages = merged_data

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar([f"${tip_ranges[i - 1]}-{tip_ranges[i]}" for i in range(1, len(tip_ranges))], tip_percentages[:-1])
    plt.xlabel('Tip Range (USD)')
    plt.ylabel('Rack Time')
    plt.title('Distribution of Tips in Different Price Ranges')
    plt.xticks(rotation=45)
    plt.show()

def visualize_predictions(X_test, y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['subtotal_amount_USD'], y_test, color='blue', label='Actual')
    plt.scatter(X_test['subtotal_amount_USD'], predictions, color='red', label='Predicted')
    plt.xlabel('Subtotal Amount (USD)')
    plt.ylabel('Tip (USD)')
    plt.title('Actual vs. Predicted Tips 50:50:0 (GoodTip:BadTip:ZeroTip)')
    plt.legend()
    plt.show()



def main(visualize=True, save_artifacts=False):
    order_data, store_data = load_data()
    merged_data = preprocess_data(order_data, store_data, tip_percentage = 0.12)
    X_train, X_test, y_train, y_test = data_loader(merged_data, test_size=0.2, 
                                                   percentage_zero_dollar_tip=0.2)
                                                   
    model, accuracy = train_model(X_train, X_test, y_train, y_test)
    print(f'Model Accuracy: {accuracy:.2f}')
    logging.info(f'Model Accuracy: {accuracy:.2f}')


    if save_artifacts:
        # Save merged_data to a CSV file with the specified naming convention
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'merged_data_{current_datetime}.csv'
        merged_data.to_csv(file_name, index=False)
        print(f"Merged data saved as '{file_name}'.")

    if visualize:
        visualize_tip_distribution(merged_data)
        # visualize_correlation(merged_data)
        predictions = model.predict(X_test)
        visualize_predictions(X_test, y_test, predictions)
        # Additional visualization for linear regression can be added here.

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process data and perform analysis.')
    parser.add_argument('-V', '--visualize', action='store_true', help='Visualize correlation graphs.')
    parser.add_argument('-A', '--artifacts', action='store_true', help='Save merged data to CSV.')
    args = parser.parse_args()

    main(visualize=args.visualize, save_artifacts=args.artifacts)
