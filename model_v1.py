# pip install pandas scikit-learn seaborn matplotlib
import os
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import statsmodels.api as sm
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from sklearn.model_selection import KFold


mpl.use('TkAgg')


class TerminalColors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


# Set up logging configuration
logging.basicConfig(filename='Data/log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# load_data() Function:
# Simple function to allow for better organization of code.
# Input: Strings of the order and store data source locations
# Output: The pandas instance of each file for Python manipulation
def load_data(order_source, store_source):
    order_data = pd.read_csv(order_source)
    store_data = pd.read_csv(store_source)
    return order_data, store_data


# preprocess_data Function():
# Processes the original data, creating a merged dataset and also calcuates an order total with tax (total_amount_USD).
# Input: raw order data, raw store data, and the definition of a 'good tip'
# (i.e., 12% or more is a good tip, 0-12% is bad, etc.)
# Output: Processed (merged) dataset
def preprocess_data(order_data, store_data, tip_percentage, percent_zero):
    # Remove non-delivery orders
    order_data = order_data[order_data['Destination_type'] == 'Delivery']

    # Remove non-dispatch orders
    order_data = order_data[order_data['Destination_Channel'] == 'Dispatch']

    # Remove 3rd party aggregators as sources (this allows us to see tip data for deliveries going to marketplace)
    order_data = order_data[~order_data['Source_actor'].isin(
        ['Uber Eats', 'DoorDash', 'Grubhub'])]

    # Remove data with negative tips
    order_data = order_data[order_data['Tip_USD'] >= 0]

    # Remove data with negative or null rack time
    order_data = order_data[~order_data['Rack_time'].isnull()]
    order_data = order_data[order_data['Rack_time'] >= 0]

    # Remove data with Tip_USD values above $75
    order_data = order_data[order_data['Tip_USD'] <= 75]

    # Merge two datapoints using Store_dma_id as primary_key
    merged_data = pd.merge(order_data, store_data,
                           on='Store_dma_id', how='inner')

    # Sum post-tax
    merged_data['total_amount_USD'] = (
        merged_data['total_tax_USD'] + merged_data['net_sales_USD'])

    # Determine if the tipped amount is considered a good tip or not
    merged_data['good_tip'] = merged_data.apply(
        lambda row: 'TRUE'
        if row['Tip_USD'] > tip_percentage * row['total_amount_USD']
        else ('ZERO' if row['Tip_USD'] == 0 else 'FALSE'), axis=1)

    # List of columns to one-hot encode
    columns_to_encode = ['store_number', 'Store_postal_code', 'Store_zip4_code',
                         'Businesses', 'Store_locale_name']

    # Perform one-hot encoding
    merged_data = pd.get_dummies(merged_data, columns=columns_to_encode)

    # Log data statistics
    total_rows = len(merged_data)
    good_tip_percentage = len(
        merged_data[merged_data['good_tip'] == 'TRUE']) / total_rows * 100
    bad_tip_percentage = len(
        merged_data[merged_data['good_tip'] == 'FALSE']) / total_rows * 100
    zero_tip_percentage = len(
        merged_data[merged_data['good_tip'] == 'ZERO']) / total_rows * 100
    print(f"Percentage of good tip data: {good_tip_percentage:.2f}%")
    print(f"Percentage of bad tip data: {bad_tip_percentage:.2f}%")
    print(f"Percentage of zero tip data: {zero_tip_percentage:.2f}%")
    print("data size", merged_data.size)
    logging.info(f"Percentage of good tip data: {good_tip_percentage:.2f}%")
    logging.info(f"Percentage of bad tip data: {bad_tip_percentage:.2f}%")
    logging.info(f"Percentage of zero tip data: {zero_tip_percentage:.2f}%")
    logging.info("data size %d", merged_data.size)

    # Throw an error if the percent of zero dollar tips requested in dataset is higher than in the processed dataset
    if percent_zero > zero_tip_percentage:
        raise ValueError(
            f"Error! Percent zero tips required by training is greater than percentage zero tips in raw dataset:\n\t"
            f"{percent_zero} > {zero_tip_percentage} -> {percent_zero > zero_tip_percentage}")

    return merged_data


# TODO: find the test train ratio
def data_loader(data):
    # Features for predicting rack time (starting with default columns)
    features = ['total_amount_USD', 'Tip_USD', 'Store_dma_id']
    # Identify columns with specific prefixes and add them to the features list
    for column in data.columns:
        if column.startswith('Businesses') or column.startswith('Store_postal_code') \
            or column.startswith('Store_zip4_code') or \
                column.startswith('Store_locale_name') or column.startswith('store_number'):
            features.append(column)

    # Select features and target variable (Rack_time)
    X = data[features]
    y = data['Rack_time']

    print("Input data dimensions (samples, features):",
          X.shape)  # Print input data dimensions

    n_splits = 10  # number of folds

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, X_test, y_train, y_test, features


# train_linear_model() Function:
# Train the linear regression model, then test based on trained model predictions.
# Accuracy is calculated in this function.
# Input: Data instances for the inputs and outputs of the model testing variables
# Output: The (now trained and testd) model and it's accuracy
def train_linear_regression(X_train, y_train, X_test, y_test):
    # TODO: add bias?
    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"{TerminalColors.GREEN}Train MSE: {train_mse}{TerminalColors.END}")
    print(f"{TerminalColors.GREEN}Test MSE: {test_mse}{TerminalColors.END}")

    return model, y_test_pred


def create_prediction_csv(X_test, y_test, predictions, prefix):
    file_name = prefix + 'predictions.csv'
    results = pd.DataFrame(
        {'Total_Amount_USD': X_test['total_amount_USD'], 'Actual_Tip_USD': y_test, 'Predicted_Tip_USD': predictions})
    results.to_csv(file_name, index=False)
    print(f"Predictions saved to '{file_name}'.")


# visualize_correlation() Function:
# Generate data visualizations for correlation of tips and rack time.
# Input: Preprocessed data (merged data)
# Output: Generated data visualization of tips and their (bucketed) correlation to rack time in the dataset
def visualize_correlation(merged_data):
    # calculate the average rack time when the tip is zero
    avg_rack_time_zero_tip = merged_data[merged_data['Tip_USD'] == 0]['Rack_time'].mean(
    )
    print(
        f"Average Rack Time when Tip is Zero: {avg_rack_time_zero_tip:.2f} minutes")

    # calculate the average rack time when the tip is more than zero
    avg_rack_time_non_zero_tip = merged_data[merged_data['Tip_USD'] > 0]['Rack_time'].mean(
    )
    print(
        f"Average Rack Time when Tip is More than Zero: {avg_rack_time_non_zero_tip:.2f} minutes")

    # scatterplot before removing zero-dollar tips
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    sns.scatterplot(y='Rack_time', x='Tip_USD', data=merged_data)
    plt.ylabel('Rack Time (minutes)')
    plt.xlabel('Tip (USD)')
    plt.title('Rack Time vs. Tip With Statistical Outliers')
    plt.ylim(1, 50)
    plt.xlim(0, 50)

    # # Line chart before removing zero-dollar tips
    plt.subplot(2, 2, 2)
    sns.lineplot(y='Rack_time', x='Tip_USD', data=merged_data)
    plt.ylabel('Rack Time')
    plt.xlabel('Tip (USD)')
    plt.title('Correlation between Rack Time and Tip (Line Chart) - Before Removal')
    plt.ylim(1, 50)
    plt.xlim(0, 50)

    # show statistics before removal
    zero_dollar_tips_percentage_before = (
        len(merged_data[merged_data['Tip_USD'] == 0]) / len(merged_data)) * 100
    correlation_coefficient_before = merged_data['Tip_USD'].corr(
        merged_data['Rack_time'])
    number_of_data_points_before = len(merged_data)
    print(
        f"Percentage of $0 tips (Before): {zero_dollar_tips_percentage_before:.2f}%")
    print(
        f"Correlation coefficient (Pearson) between Rack Time and Tip (Before): {correlation_coefficient_before:.2f}")
    print(f"Number of data points (Before): {number_of_data_points_before}")

    # remove outliers (tip values over $30)
    merged_data = merged_data[merged_data['Tip_USD'] <= 30]

    # scatterplot after removing zero-dollar tips and outliers
    plt.subplot(2, 2, 3)
    sns.scatterplot(y='Rack_time', x='Tip_USD', data=merged_data)
    plt.ylabel('Rack Time (minutes)')
    plt.xlabel('Tip (USD)')
    plt.title('Rack Time vs. Tip Without Statistical Outliers')
    plt.ylim(1, 50)
    plt.xlim(0, 30)  # Updated ylim to accommodate the removal of outliers

    # # line chart after removing zero-dollar tips and outliers
    plt.subplot(2, 2, 4)
    sns.lineplot(y='Rack_time', x='Tip_USD', data=merged_data)
    plt.ylabel('Rack Time')
    plt.xlabel('Tip (USD)')
    plt.title('Correlation between Rack Time and Tip (Line Chart) - After Removal')
    plt.ylim(1, 50)
    plt.xlim(0, 30)  # Updated ylim to accommodate the removal of outliers

    # calculate statistics after removal
    zero_dollar_tips_percentage_after = (
        len(merged_data[merged_data['Tip_USD'] == 0]) / len(merged_data)) * 100
    correlation_coefficient_after = merged_data['Tip_USD'].corr(
        merged_data['Rack_time'])
    number_of_data_points_after = len(merged_data)
    print(
        f"Percentage of $0 tips (After Removal): {zero_dollar_tips_percentage_after:.2f}%")
    print(
        f"Correlation coefficient (Pearson) between Rack Time and Tip (After Removal): {correlation_coefficient_after:.2f}")
    print(
        f"Number of data points (After Removal): {number_of_data_points_after}")

    plt.tight_layout()
    plt.show()


# visualize_tip_distribution() Function:
# Generate data visualizations for data distribution based on tips.
# Input: Preprocessed data (merged data)
# Output: Generated data visualization of the distribution of tips in the dataset
def visualize_tip_distribution(merged_data):
    # Tip ranges
    tip_ranges = [0, 3, 5, 8, 12, 15, 30, float('inf')]

    # Initialize a dictionary to store Rack_time sums (in minutes) for each tip range
    rack_time_sums = {
        f"${tip_ranges[i - 1]}-{tip_ranges[i]}": 0 for i in range(1, len(tip_ranges))}

    # Count Rack_time in minutes for each tip range
    for index, row in merged_data.iterrows():
        tip = row['Tip_USD']
        # Convert Rack_time from seconds to minutes
        rack_time_minutes = row['Rack_time'] / 60
        # Skip the first range (0-1)
        for i, max_tip in enumerate(tip_ranges[1:]):
            if tip <= max_tip:
                tip_range_key = f"${tip_ranges[i]}-{max_tip}"
                rack_time_sums[tip_range_key] += rack_time_minutes
                break
        else:
            tip_range_key = f"${tip_ranges[-2]}-{tip_ranges[-1]}"
            rack_time_sums[tip_range_key] += rack_time_minutes

    # Extract Rack_time sums and tip range keys for plotting
    rack_time_values = list(rack_time_sums.values())
    tip_range_labels = list(rack_time_sums.keys())

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(tip_range_labels, rack_time_values)
    plt.xlabel('Tip Range (USD)')
    plt.ylabel('Rack Time (minutes)')
    plt.title('Rack Time Distribution in Different Tip Ranges')
    plt.xticks(rotation=45)
    plt.show()


# visualize_predictions() Function:
# Generate data visualization for linear regression model predictions compared to actual values
# This is a method of checking accuracy and determining optimal tip vs. actual tip
# Input: Linear regression model test data, and the predicted data
# Output: Generated data visualization
def visualize_predictions(X_test, y_test, predictions, accuracy, good_tip, percent_good, percent_bad, percent_zero):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['total_amount_USD'], y_test,
                color='blue', label='Actual')
    plt.scatter(X_test['total_amount_USD'], predictions,
                color='red', label='Optimal')
    plt.xlabel('Total Amount (USD)')
    plt.ylabel('Tip (USD)')
    plt.title(
        f'Actual vs. Predicted Tips {percent_good * 100}:{percent_bad * 100}:{percent_zero * 100} (Good:Bad:Zero)')
    plt.legend()

    # Add important background information to the plot
    plot_annotation = f'*Good tip considered to be {good_tip * 100}% or higher.\nAccuracy: {(accuracy*100):.2f}%'
    plt.text(1, -0.1, plot_annotation, fontsize=9, ha='right',
             va='center', transform=plt.gca().transAxes)

    # Add the equation of the predicted line
    coefs = np.polyfit(X_test['total_amount_USD'], predictions, 1)
    equation = f"Predicted Line: y = {coefs[0]:.2f}x + {coefs[1]:.2f}"
    plt.text(-0.015, -0.08, equation, fontsize=9, ha='left',
             va='center', transform=plt.gca().transAxes)

    plt.show()


# main() Function:
# Executes the program based on input arguments and relative data
# Defaults to visualize (-V argument) and not save generated artifacts (-A argument) such as:
# - Generated processed data CSVs
# - Data visualization files
# These defaults can be changed with input arguments in the command line.
def main(visualize=True, save_artifacts=False):
    # Define Order and Store data CSV file locations
    order_source = 'Data/dispatch_order_data.csv'
    store_source = 'Data/dispatch_store_data.csv'

    # Define the distribution of good tips to bad tips to zero dollar tips for the model training
    percent_good = 0.6
    percent_bad = 0.3
    percent_zero = 0.1

    # Define what a good tip, like what is an ideal tip.
    # (i.e., 0.12 would mean 12+% is considered good)
    good_tip_definition = 0.12

    # Confirm if the defined distribution of tips for model training is divided correctly
    if round(percent_good + percent_bad + percent_zero, 4) != 1:
        raise ValueError(
            f"Error! Good:Bad:Zero distribution does not equal 100%... it equals {percent_good + percent_bad + percent_zero}")

    # Load pandas data file instances
    order_data, store_data = load_data(order_source, store_source)

    # Preprocess data, then load the model data instances for the training and testing
    merged_data = preprocess_data(
        order_data, store_data, tip_percentage=good_tip_definition, percent_zero=percent_zero)
    X_train, X_test, y_train, y_test, features = data_loader(merged_data)
    visualize_tip_distribution(merged_data)
    lr_model, lr_predictions = train_linear_regression(
        X_train, y_train, X_test, y_test)

    # Coefficients/Weights (w1, w2, w3)
    coefficients = lr_model.coef_
    print(f"{TerminalColors.GREEN}Coefficients/Weights:",
          coefficients, f"{TerminalColors.END}")
    data = {'Feature Name': features, 'Coefficient Value': coefficients}
    df = pd.DataFrame(data)

    # Export to an Excel file
    df.to_csv('features_coefficients.csv', index=False)
    # Intercept (bias)
    intercept = lr_model.intercept_
    print("Intercept (bias):", intercept)

    threshold_predicted_rack_time = 7.0

    if save_artifacts:
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'merged_data_{current_datetime}.csv'
        merged_data.to_csv(file_name, index=False)
        print(f"Merged data saved as '{file_name}'.")

    # Show visualization for raw data and model predictions through matplotlib graphs
    if visualize:
        # visualize the distribution of tips in processed data
        visualize_tip_distribution(merged_data)

        # visualize the correlation of tips to rack time in raw data
        visualize_correlation(merged_data)

        # predict how much should be tipped for an order
        # predictions = model.predict(X_test)

        # # visualize the accuracy of model predictions
        # visualize_predictions(X_test, y_test, predictions, accuracy, good_tip_definition, percent_good, percent_bad, percent_zero)

        # Additional visualization for linear regression and neural network to be added here.


# Add possible arguments for Python execution.
# '-V' argument: Visualize the data for linear regression and neural network models
# '-A' argument: Save any generated artifacts. Only CSVs at this point, to automatically
#                save any data visualizations eventually.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Process data and perform analysis.')
    parser.add_argument('-V', '--visualize', action='store_true',
                        help='Visualize correlation graphs.')
    parser.add_argument('-A', '--artifacts',
                        action='store_true', help='Save merged data to CSV.')
    args = parser.parse_args()

    main(visualize=args.visualize, save_artifacts=args.artifacts)
