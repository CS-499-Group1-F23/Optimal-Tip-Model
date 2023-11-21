# pip install pandas scikit-learn seaborn matplotlib
import os
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import KFold

mpl.use('TkAgg')

# Predefined hyperparameters (modifiable)
num_filters = 64
kernel_size = 3
pool_size = 2
dropout_rate = 0.5
dense_units = 64
learning_rate = 0.0001
batch_size = 32
epochs = 10

def create_cnn(input_shape):
    model = Sequential([
        Conv1D(num_filters, kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=pool_size),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)  # Output layer for regression
    ])
    return model


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
    order_data = order_data[~order_data['Source_actor'].isin(['Uber Eats', 'DoorDash', 'Grubhub'])]

    # Remove data with negative tips
    order_data = order_data[order_data['Tip_USD'] >= 0]

    # Remove data with negative or null rack time
    order_data = order_data[~order_data['Rack_time'].isnull()]
    order_data = order_data[order_data['Rack_time'] >= 0]

    # Remove data with Tip_USD values above $75
    order_data = order_data[order_data['Tip_USD'] <= 75]

    # Merge two datapoints using store_number as primary_key
    merged_data = pd.merge(order_data, store_data, on='store_number', how='inner')

    # Sum post-tax
    merged_data['total_amount_USD'] = (merged_data['total_tax_USD'] + merged_data['net_sales_USD'])

    # Determine if the tipped amount is considered a good tip or not
    merged_data['good_tip'] = merged_data.apply(
        lambda row: 'TRUE'
        if row['Tip_USD'] > tip_percentage * row['total_amount_USD']
        else ('ZERO' if row['Tip_USD'] == 0 else 'FALSE')
        , axis=1)
    
        # List of columns to one-hot encode
    columns_to_encode = ['Store_dma_id_x', 'store_number', 'Store_postal_code', 'Store_zip4_code',
                         'Businesses', 'Store_locale_name']

    # Perform one-hot encoding
    merged_data = pd.get_dummies(merged_data, columns=columns_to_encode)

    # Log data statistics
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

    # Throw an error if the percent of zero dollar tips requested in dataset is higher than in the processed dataset
    if percent_zero > zero_tip_percentage:
        raise ValueError(
            f"Error! Percent zero tips required by training is greater than percentage zero tips in raw dataset:\n\t"
            f"{percent_zero} > {zero_tip_percentage} -> {percent_zero > zero_tip_percentage}")

    return merged_data


# data_loader Function()
# Load data instances to be fed to the model for training and testing based on the merged data.
# Input: Preprocessed (merged) data, test size, percent of zero tips, percent bad tips, percent good tips
# Output: Data instances for model training and testing
def data_loader_old(merged_data, test_size, percentage_zero_dollar_tip, percentage_bad_tip=None, percentage_good_tip=None):
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

    X = merged_data[['store_number', 'total_amount_USD', 'Rack_time']]
    y = merged_data['Tip_USD']

    n_splits= 10 # number of folds

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(X_test.size)
    return X_train, X_test, y_train, y_test

def data_loader(data):
    # Features for predicting rack time (starting with default columns)
    features = [
        'total_amount_USD', 'Tip_USD', 'Area_sqmi', 'Avg_income']
    features.sort()
    # Identify columns with specific prefixes and add them to the features list
    for column in data.columns:
        if column.startswith('Businesses') or column.startswith('Store_postal_code') \
                or column.startswith('Store_zip4_code') or column.startswith('Store_dma_id')or column.startswith('Store_locale_name'):
            features.append(column)

    # Select features and target variable (Rack_time)
    X = data[features]
    y = data['Rack_time']
    
    print("Input data dimensions (samples, features):", X.shape)  # Print input data dimensions

    n_splits= 10 # number of folds

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return X_train, X_test, y_train, y_test, features


def tip_data_loader(data, threshold_rack_time):
    # Filter data for predicted rack times above the threshold
    data_filtered = data[data['predicted_rack_time'] > threshold_rack_time]

    # Features for predicting tips
    tip_features = [
        'total_amount_USD', 'Area_sqmi', 'Avg_income','predicted_rack_time']
    for column in data.columns:
        if column.startswith('Businesses') or column.startswith('Store_postal_code') \
            or column.startswith('Store_zip4_code') or column.startswith('Store_dma_id') \
            or column.startswith('Store_locale_name'):
            tip_features.append(column)

    X = data_filtered[tip_features]
    y = data_filtered['Tip_USD']  # Target variable: Tip amount

    n_splits = 10

    kf = KFold(n_splits= n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    # TODO: add bias?
    # Initialize and train the linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    return lr_model

def train_cnn(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    input_shape = (X_train_scaled.shape[1], 1)  # Define input shape for CNN
    
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    model = create_cnn(input_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    # history = model.fit(X_train_scaled, y_train, batch_size=batch_size, epochs=epochs, 
    #                     validation_data=(X_test_scaled, y_test), verbose=1)
    # model.fit(X_train_scaled, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test_scaled, y_test))
    #  # Train the model
    # print("Loss function history during training:")
    # print(history.history['loss']) 
    history = model.fit(X_train_scaled, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test_scaled, y_test))
    
    # Print final loss after training
    final_loss = model.evaluate(X_test_scaled, y_test)
    print(f"{TerminalColors.GREEN}Final Loss CNN (MSE): {final_loss}{TerminalColors.END}")
    return model


# train_linear_model() Function:
# Train the linear regression model, then test based on trained model predictions.
# Accuracy is calculated in this function.
# Input: Data instances for the inputs and outputs of the model testing variables
# Output: The (now trained and testd) model and it's accuracy
def train_linear_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions, squared=False)
    accuracy = 1 - (mse / y_test.var())
    return model, accuracy

# Function to train a Random Forest model
n_estimators = 100
max_depth = None  # Change this to a specific depth if needed

def train_random_forest(X_train, X_test, y_train, y_test):
    # Initialize Random Forest Regressor with predefined hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    # Fit the model
    model.fit(X_train, y_train)
    # Evaluate on test set and print the loss function (MSE)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"{TerminalColors.GREEN}Final Loss RF (MSE): {mse}{TerminalColors.END}")
    # Access the tree depths
    tree_depths = [estimator.tree_.max_depth for estimator in model.estimators_]
    print(f"Tree Depths RF: {tree_depths}")
    return model


# Function to train a Gradient Boosting model
n_estimators = 100
max_depth = 3
learning_rate = 0.001
def train_gradient_boosting(X_train, X_test, y_train, y_test):
    # Initialize Gradient Boosting Regressor with predefined hyperparameters
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Evaluate on test set and print the loss function (MSE)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"{TerminalColors.GREEN}Final Loss GB (MSE): {mse}{TerminalColors.END}")
    return model


def create_prediction_csv(X_test, y_test, predictions, prefix):
    file_name = prefix + 'predictions.csv'
    results = pd.DataFrame({'Total_Amount_USD': X_test['total_amount_USD'], 'Actual_Tip_USD': y_test, 'Predicted_Tip_USD': predictions})
    results.to_csv(file_name, index=False)
    print(f"Predictions saved to '{file_name}'.")


# visualize_correlation() Function:
# Generate data visualizations for correlation of tips and rack time.
# Input: Preprocessed data (merged data)
# Output: Generated data visualization of tips and their (bucketed) correlation to rack time in the dataset
def visualize_correlation(merged_data):
    # calculate the average rack time when the tip is zero
    avg_rack_time_zero_tip = merged_data[merged_data['Tip_USD'] == 0]['Rack_time'].mean()
    print(f"Average Rack Time when Tip is Zero: {avg_rack_time_zero_tip:.2f} minutes")

    # calculate the average rack time when the tip is more than zero
    avg_rack_time_non_zero_tip = merged_data[merged_data['Tip_USD'] > 0]['Rack_time'].mean()
    print(f"Average Rack Time when Tip is More than Zero: {avg_rack_time_non_zero_tip:.2f} minutes")

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
    # plt.subplot(2, 2, 2)
    # sns.lineplot(y='Rack_time', x='Tip_USD', data=merged_data)
    # plt.ylabel('Rack Time')
    # plt.xlabel('Tip (USD)')
    # plt.title('Correlation between Rack Time and Tip (Line Chart) - Before Removal')
    # plt.ylim(1, 50)
    # plt.xlim(0, 50)

    # show statistics before removal
    zero_dollar_tips_percentage_before = (len(merged_data[merged_data['Tip_USD'] == 0]) / len(merged_data)) * 100
    correlation_coefficient_before = merged_data['Tip_USD'].corr(merged_data['Rack_time'])
    number_of_data_points_before = len(merged_data)
    print(f"Percentage of $0 tips (Before): {zero_dollar_tips_percentage_before:.2f}%")
    print(f"Correlation coefficient (Pearson) between Rack Time and Tip (Before): {correlation_coefficient_before:.2f}")
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
    # plt.subplot(2, 2, 4)
    # sns.lineplot(y='Rack_time', x='Tip_USD', data=merged_data)
    # plt.ylabel('Rack Time')
    # plt.xlabel('Tip (USD)')
    # plt.title('Correlation between Rack Time and Tip (Line Chart) - After Removal')
    # plt.ylim(1, 50)
    # plt.xlim(0, 30)  # Updated ylim to accommodate the removal of outliers

    # calculate statistics after removal
    zero_dollar_tips_percentage_after = (len(merged_data[merged_data['Tip_USD'] == 0]) / len(merged_data)) * 100
    correlation_coefficient_after = merged_data['Tip_USD'].corr(merged_data['Rack_time'])
    number_of_data_points_after = len(merged_data)
    print(f"Percentage of $0 tips (After Removal): {zero_dollar_tips_percentage_after:.2f}%")
    print(f"Correlation coefficient (Pearson) between Rack Time and Tip (After Removal): {correlation_coefficient_after:.2f}")
    print(f"Number of data points (After Removal): {number_of_data_points_after}")


    plt.tight_layout()
    plt.show()


# visualize_tip_distribution() Function:
# Generate data visualizations for data distribution based on tips.
# Input: Preprocessed data (merged data)
# Output: Generated data visualization of the distribution of tips in the dataset
def visualize_tip_distribution(merged_data):
    # Tip ranges
    tip_ranges = [0, 1, 3, 5, 8, 12, 15, 30, float('inf')]

    # Initialize a dictionary to store Rack_time sums (in minutes) for each tip range
    rack_time_sums = {f"${tip_ranges[i - 1]}-{tip_ranges[i]}": 0 for i in range(1, len(tip_ranges))}

    # Count Rack_time in minutes for each tip range
    for index, row in merged_data.iterrows():
        tip = row['Tip_USD']
        rack_time_minutes = row['Rack_time'] / 60  # Convert Rack_time from seconds to minutes
        for i, max_tip in enumerate(tip_ranges[1:]):  # Skip the first range (0-1)
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
    plt.scatter(X_test['total_amount_USD'], y_test, color='blue', label='Actual')
    plt.scatter(X_test['total_amount_USD'], predictions, color='red', label='Optimal')
    plt.xlabel('Total Amount (USD)')
    plt.ylabel('Tip (USD)')
    plt.title(f'Actual vs. Predicted Tips {percent_good * 100}:{percent_bad * 100}:{percent_zero * 100} (Good:Bad:Zero)')
    plt.legend()
    
    # Add important background information to the plot
    plot_annotation = f'*Good tip considered to be {good_tip * 100}% or higher.\nAccuracy: {(accuracy*100):.2f}%'
    plt.text(1, -0.1, plot_annotation, fontsize=9, ha='right', va='center', transform=plt.gca().transAxes)

    # Add the equation of the predicted line
    coefs = np.polyfit(X_test['total_amount_USD'], predictions, 1)
    equation = f"Predicted Line: y = {coefs[0]:.2f}x + {coefs[1]:.2f}"
    plt.text(-0.015, -0.08, equation, fontsize=9, ha='left', va='center', transform=plt.gca().transAxes)

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

    # Define the test size in percent of original data set
    # (i.e., 0.2 would mean 20% of the data set is used for testing, 80% is used for training)
    test_size = 0.2

    # Confirm if the defined distribution of tips for model training is divided correctly
    if round(percent_good + percent_bad + percent_zero, 4) != 1:
        raise ValueError(f"Error! Good:Bad:Zero distribution does not equal 100%... it equals {percent_good + percent_bad + percent_zero}")

    # Load pandas data file instances
    order_data, store_data = load_data(order_source, store_source)

    # Preprocess data, then load the model data instances for the training and testing
    merged_data = preprocess_data(order_data, store_data, tip_percentage=good_tip_definition, percent_zero=percent_zero)
    X_train, X_test, y_train, y_test, features_for_rack_time_prediction  = data_loader(merged_data)
    # X_train, X_test, y_train, y_test = data_loader(merged_data,
    #                                                test_size=test_size,
    #                                                percentage_good_tip=percent_good,
    #                                                percentage_bad_tip=percent_bad,
    #                                                percentage_zero_dollar_tip=percent_zero)
    y_train_log = np.log1p(y_train)  # Logarithmic transformation for training set
    y_test_log = np.log1p(y_test)
    log_model = LinearRegression()
    log_model.fit(X_train, y_train_log)
    predictions_log = log_model.predict(X_test)
    predictions = np.expm1(predictions_log)  # Inverse transformation from logarithmic scale
    mse = mean_squared_error(y_test, predictions)
    model = train_linear_regression(X_train, y_train)
    print(f"{TerminalColors.GREEN}Log Mean Squared Error: {mse}{TerminalColors.END}")
    threshold_value_log = np.median(predictions)  # Using median as the threshold value for logarithmic predictions
    print(f"Threshold Value based on Predicted Rack Time (using logarithmic regression): {threshold_value_log}")

    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, predictions)
    print(f"{TerminalColors.GREEN}RACK Time Mean Squared Error: {mse}{TerminalColors.END}")

    # Calculate threshold value based on the predicted rack time
    threshold_value = predictions.mean()  # Using mean as a simple threshold example
    print(f"Threshold Value based on Predicted Rack Time (using mean): {threshold_value}")
    threshold_value = np.median(predictions)  # Using median as the threshold value
    print(f"Threshold Value based on Predicted Rack Time (using median): {threshold_value}")
    feature_weights = model.coef_
    feature_names = X_train.columns  
    feature_weights_df = pd.DataFrame({'Feature': feature_names, 'Weight': feature_weights})
    print(feature_weights_df)

    trained_model = train_cnn(X_train, X_test, y_train, y_test)
    layers = trained_model.layers

    # # Extracting weights of the layers
    # for layer in layers:
    #     if hasattr(layer, 'get_weights'):
    #         weights = layer.get_weights()
    #         if weights:
    #             print(f"Weights for {layer.name}: {weights}")

    merged_data['predicted_rack_time'] = model.predict(merged_data[features_for_rack_time_prediction])
    merged_data.to_csv("test.csv", index=False)
    X_train_tip, X_test_tip, y_train_tip, y_test_tip = tip_data_loader(merged_data, threshold_value)
    
    gb_model = train_gradient_boosting(X_train, X_test, y_train, y_test)
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    tip_model = train_linear_regression(X_train_tip, y_train_tip)
    # Predict on the test set
    tip_predictions = tip_model.predict(X_test_tip)
    tip_feature_weights = tip_model.coef_
    tip_feature_names = X_train_tip.columns  # Assuming X_train is your training feature set
    feature_weights_df = pd.DataFrame({'Feature': tip_feature_names, 'Weight': tip_feature_weights})
    print(feature_weights_df)

    # Calculate MSE for tip predictions
    tip_mse = mean_squared_error(y_test_tip, tip_predictions)
    print(f"{TerminalColors.GREEN}Mean Squared Error for Tip Prediction: {tip_mse}{TerminalColors.END}")
    tip_trained_model = train_cnn( X_train_tip, X_test_tip, y_train_tip, y_test_tip)
    tip_gb_model = train_gradient_boosting(X_train_tip, X_test_tip, y_train_tip, y_test_tip)
    tip_rf_model = train_random_forest( X_train_tip, X_test_tip, y_train_tip, y_test_tip)



    # # Train the model, log relative accuracy with the given input data
    # model, accuracy = train_linear_model(X_train, X_test, y_train, y_test)
    
    # # predict how much should be tipped for an order
    # predictions = model.predict(X_test)

    # create_prediction_csv(X_test, y_test, predictions, prefix="lr")

    # print(f'Model Accuracy: {accuracy:.2f}')
    # logging.info(f'Model Accuracy: {accuracy:.2f}')

    # model_rf, accuracy_rf, predictions_rf = train_random_forest_model(X_train, X_test, y_train, y_test)
    # print(f'Random Forest Model Accuracy: {accuracy_rf:.2f}')
    # logging.info(f'Random Forest Model Accuracy: {accuracy_rf:.2f}')

    # create_prediction_csv(X_test, y_test, predictions_rf, prefix = "rf")

    # # Train the Gradient Boosting model
    # model_gb, accuracy_gb, predictions_gb = train_gradient_boosting_model(X_train, X_test, y_train, y_test)
    # print(f'Gradient Boosting Model Accuracy: {accuracy_gb:.2f}')
    # logging.info(f'Gradient Boosting Model Accuracy: {accuracy_gb:.2f}')
    # create_prediction_csv(X_test, y_test, predictions_gb, prefix = "gb")

    # Save merged_data to a CSV file with the specified naming convention
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
        predictions = model.predict(X_test)

        # visualize the accuracy of model predictions
        visualize_predictions(X_test, y_test, predictions, accuracy, good_tip_definition, percent_good, percent_bad, percent_zero)

        # Additional visualization for linear regression and neural network to be added here.


# Add possible arguments for Python execution.
# '-V' argument: Visualize the data for linear regression and neural network models
# '-A' argument: Save any generated artifacts. Only CSVs at this point, to automatically
#                save any data visualizations eventually.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process data and perform analysis.')
    parser.add_argument('-V', '--visualize', action='store_true', help='Visualize correlation graphs.')
    parser.add_argument('-A', '--artifacts', action='store_true', help='Save merged data to CSV.')
    args = parser.parse_args()

    main(visualize=args.visualize, save_artifacts=args.artifacts)
