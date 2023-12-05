# pip install datetime logging pandas joblib scikit-learn seaborn matplotlib tensorflow statsmodels numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import joblib
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import KFold

mpl.use('TkAgg')
class TerminalColors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @classmethod
    def bold_text(cls, text):
        return cls.BOLD + text + cls.END
    
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
    print(f"{TerminalColors.YELLOW + TerminalColors.BOLD}Preprocessing data...{TerminalColors.END}")
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
    columns_to_encode = ['Store_postal_code', 'Store_zip4_code',
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
    print(f"{TerminalColors.YELLOW}Percentage of good tip data: {good_tip_percentage:.2f}%")
    print(f"Percentage of bad tip data: {bad_tip_percentage:.2f}%")
    print(f"Percentage of zero tip data: {zero_tip_percentage:.2f}%")
    print(f"{TerminalColors.END}\n")
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

def data_loader(data):
    print(f"{TerminalColors.RED + TerminalColors.BOLD}Loading data...{TerminalColors.END}")
    # Features for predicting rack time (starting with default columns)
    features = ['total_amount_USD', 'Tip_USD', 'Area_sqmi']
    # Identify columns with specific prefixes and add them to the features list
    # for column in data.columns:
    #     if column.startswith('Store_dma_id'):
    #         features.append(column)

    # Select features and target variable (Rack_time)
    X = data[features]
    y = data['Rack_time']

    print(f"{TerminalColors.RED}Input data dimensions (samples, features):",
          X.shape)  # Print input data dimensions

    n_splits = 10  # number of folds

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    print(f"{TerminalColors.RED}Input data dimensions (samples, features):", X.shape) 
    print("Train set dimensions (samples, features):", X_train.shape)
    print("Test set dimensions (samples, features):", X_test.shape)
    print(f"{TerminalColors.END}\n")
    return X_train, X_test, y_train, y_test, features


"""
Trains and evaluates a linear regression model.

Trains the model with X_train and y_train, then predicts and calculates MSE on X_test and y_test. 
Model's coefficients, intercept, and MSE are printed. The model is saved to a file.

Parameters:
X_train, y_train: Training data.
X_test, y_test: Testing data.

Returns:
Trained model and its predictions on test data.
"""
def train_linear_regression(X_train, y_train, X_test, y_test):
    print(f"{TerminalColors.GREEN + TerminalColors.BOLD}Training linear regression Training{TerminalColors.END}")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    coefficients = model.coef_
    features = X_train.columns.tolist()
    print(f"{TerminalColors.GREEN}Features (Linear Regression): {features}")
    print(f"{TerminalColors.GREEN}Coefficients/Weights (Linear Regression):",
          coefficients)
    # Intercept (bias)
    intercept = model.intercept_
    print(f"Intercept (bias): {intercept} {TerminalColors.END}")
    print(f"{TerminalColors.GREEN + TerminalColors.BOLD}LinearRegression MSE: {test_mse}")
    print("\n")
    return model, y_test_pred


"""
Trains a feedforward neural network (FNN) on provided data and evaluates its performance.

Parameters:
X_train, y_train: Training data.
X_test, y_test: Testing data.

Returns:
The trained FNN model.
"""
def train_fnn(X_train, X_test, y_train, y_test):
    print(f"{TerminalColors.BLUE + TerminalColors.BOLD}Training ForwardFeed NN{TerminalColors.END}")
    # Define FNN architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer with 1 neuron for 'Rack_time'
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and return MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"{TerminalColors.BLUE} ForwardFeed NN Test MSE: {mse}{TerminalColors.END}")

    layer_weights = []
    for layer in model.layers:
        weights = layer.get_weights()  # Get weights of each layer
        layer_weights.append(weights)

    # Assuming the first layer is the input layer, get its weights
    input_layer_weights = layer_weights[0][0]
    # Assuming X_train has column names, assign weights to features
    feature_weights = dict(zip(X_train.columns,input_layer_weights.T))  # Transpose to match features with weights
    # print(f"{TerminalColors.BLUE}Feature Weights (FFNN): {feature_weights}{TerminalColors.END}")
    print(f"{TerminalColors.BLUE}Averages of weights\nTotal_amount = {np.average(feature_weights['total_amount_USD'])}, \
             \n Tip_USD = {np.average(feature_weights['Tip_USD'])} \
             \n Area_sqmi = {np.average(feature_weights['Area_sqmi'])} \
           {TerminalColors.END}")
    print(f"{TerminalColors.BLUE + TerminalColors.BOLD} ForwardFeed NN Test MSE: {mse}{TerminalColors.END}")
    print("\n")
    return model


"""
Creates a CSV file with actual and predicted values.

Parameters:
X_test (DataFrame): Test data containing features.
y_test (array-like): Actual target values from the test data.
predictions (array-like): Predicted target values.
prefix (str): Prefix for the output CSV file name.

Returns:
None
"""
def create_prediction_csv(X_test, y_test, predictions, prefix):
    file_name = prefix + 'predictions.csv'
    results = pd.DataFrame(
        {'Total_Amount_USD': X_test['total_amount_USD'], 'Actual_Tip_USD': y_test, 'Predicted_Tip_USD': predictions})
    results.to_csv(file_name, index=False)
    print(f"Predictions saved to '{file_name}'.")


"""
Generates visualizations to analyze the correlation between tips and rack time in a dataset.

The function calculates and prints average rack times for zero and non-zero tips, 
then creates scatterplots and line charts to visually represent the relationship 
between rack time and tip amount, both before and after removing statistical outliers.
It also prints statistics such as the percentage of $0 tips and the Pearson correlation 
coefficient before and after outlier removal.

Parameters:
merged_data (DataFrame): The preprocessed dataset containing 'Tip_USD' and 'Rack_time' columns.

Output:
Displays generated plots and prints relevant statistics.
"""
def visualize_correlation(merged_data):
    # Calculate the average rack time when the tip is zero
    avg_rack_time_zero_tip = merged_data[merged_data['Tip_USD'] == 0]['Rack_time'].mean(
    )
    print(
        f"Average Rack Time when Tip is Zero: {avg_rack_time_zero_tip:.2f} minutes")

    # Calculate the average rack time when the tip is more than zero
    avg_rack_time_non_zero_tip = merged_data[merged_data['Tip_USD'] > 0]['Rack_time'].mean(
    )
    print(
        f"Average Rack Time when Tip is More than Zero: {avg_rack_time_non_zero_tip:.2f} minutes")

    # Scatterplot before removing zero-dollar tips
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    sns.scatterplot(y='Rack_time', x='Tip_USD', data=merged_data)
    plt.ylabel('Rack Time (secounds)')
    plt.xlabel('Tip (USD)')
    plt.title('Rack Time vs. Tip With Statistical Outliers')
    plt.ylim(1, 50)
    plt.xlim(0, 50)

    # Line chart before removing zero-dollar tips
    plt.subplot(2, 2, 2)
    sns.lineplot(y='Rack_time', x='Tip_USD', data=merged_data)
    plt.ylabel('Rack Time')
    plt.xlabel('Tip (USD)')
    plt.title('Correlation between Rack Time and Tip (Line Chart) - Before Removal')
    plt.ylim(1, 50)
    plt.xlim(0, 50)

    # Show statistics before removal
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

    # Remove outliers (tip values over $30)
    merged_data = merged_data[merged_data['Tip_USD'] <= 30]

    # Scatterplot after removing zero-dollar tips and outliers
    plt.subplot(2, 2, 3)
    sns.scatterplot(y='Rack_time', x='Tip_USD', data=merged_data)
    plt.ylabel('Rack Time (secounds)')
    plt.xlabel('Tip (USD)')
    plt.title('Rack Time vs. Tip Without Statistical Outliers')
    plt.ylim(1, 50)
    plt.xlim(0, 30)

    # Line chart after removing zero-dollar tips and outliers
    plt.subplot(2, 2, 4)
    sns.lineplot(y='Rack_time', x='Tip_USD', data=merged_data)
    plt.ylabel('Rack Time')
    plt.xlabel('Tip (USD)')
    plt.title('Correlation between Rack Time and Tip (Line Chart) - After Removal')
    plt.ylim(1, 50)
    plt.xlim(0, 30)

    # Calculate statistics after removal
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


"""
Generates a bar chart visualizing the distribution of rack times across different tip ranges.

Parameters:
merged_data (DataFrame): The preprocessed dataset with 'Tip_USD' and 'Rack_time' columns.

Output:
Displays a bar chart visualizing the distribution of rack times across tip ranges.
"""
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
    plt.ylabel('Rack Time (Seconds)')
    plt.title('Rack Time Distribution in Different Tip Ranges')
    plt.xticks(rotation=45)
    plt.show()


"""
Plots actual vs. predicted tips to visualize the performance of a linear regression model.

Parameters:
X_test (DataFrame): Test data containing the 'total_amount_USD' feature.
y_test (array-like): Actual tip values.
predictions (array-like): Predicted tip values by the model.

Output:
Displays a scatter plot for actual vs. predicted tips.
"""
def visualize_predictions(X_test, y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['total_amount_USD'], y_test,
                color='blue', label='Actual')
    plt.scatter(X_test['total_amount_USD'], predictions,
                color='red', label='Optimal')
    plt.xlabel('Total Amount (USD)')
    plt.ylabel('Tip (USD)')
    plt.title('Actual vs. Predicted Tips')
    plt.legend()

    # Add the equation of the predicted line
    coefs = np.polyfit(X_test['total_amount_USD'], predictions, 1)
    equation = f"Predicted Line: y = {coefs[0]:.2f}x + {coefs[1]:.2f}"
    plt.text(-0.015, -0.08, equation, fontsize=9, ha='left',
             va='center', transform=plt.gca().transAxes)

    plt.show()


"""
Calculates optimal tips based on a threshold rack time using a trained model.

Parameters:
model (Model): A trained machine learning model.
test_input (DataFrame): Test data containing features like 'total_amount_USD' and 'Area_sqmi'.
threshold_racktime (float): The threshold rack time used in reverse engineering the model.

Returns:
List of optimal tips calculated for each instance in the test data.
"""
def get_rack_time(model, test_input, threshold_racktime):
    # Save test_input to csv
    test_input.to_csv('test_input.csv', index=False)

    # Calculate predicted rack times for the test set
    predicted_rack_times = model.predict(test_input)
    # Find the threshold rack time using the median of predicted rack times
    threshold_rack_time = np.median(predicted_rack_times)

    # Extract feature weights and intercept from the trained model
    feature_weights = model.coef_
    intercept = model.intercept_

    # Reverse engineer the model to predict Tip_USD for the test data based on weights and threshold
    optimal_tips = []
    for index, row in test_input.iterrows():
    
        # Calculate the predicted Tip_USD using the weights and threshold rack time
        predicted_tip = (threshold_rack_time - (feature_weights[0]* row['total_amount_USD']) - (row['Area_sqmi']*feature_weights[2]) - intercept) / (feature_weights[1])
        # Round the predicted tip to the nearest cent
        predicted_tip = round(predicted_tip, 2)
        optimal_tips.append(predicted_tip)
    return optimal_tips


"""
Main

Parameters:
visualize (bool): If True, visualizes data distributions and correlations.
save_artifacts (bool): If True, saves processed data to CSV files and model infrence.

Output:
Executes the data processing and modeling pipeline. Optionally saves data and displays visualizations.
"""
def main(visualize=False, save_artifacts=False):
    # Define Order and Store data CSV file locations
    order_source = ""
    store_source = ""

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
    lr_model, lr_predictions = train_linear_regression(X_train, y_train, X_test, y_test)
    train_fnn(X_train, X_test, y_train, y_test)

    get_rack_time(lr_model, X_test, threshold_racktime=7.0)
    

    if save_artifacts:
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'merged_data_{current_datetime}.csv'
        merged_data.to_csv(file_name, index=False)
        print(f"Merged data saved as '{file_name}'.")
        joblib.dump(lr_model, f'lr_model_{current_datetime}.pkl')  # Save the entire model

    # Show visualization for raw data and model predictions through matplotlib graphs
    if visualize:
        # visualize the distribution of tips in processed data
        visualize_tip_distribution(merged_data)

        # visualize the correlation of tips to rack time in raw data
        visualize_correlation(merged_data)
        

"""
 Add possible arguments for Python execution.
 '-V' argument: Visualize the data for linear regression and neural network models
 '-A' argument: Save any generated artifacts. Only CSVs at this point, to automatically
                save any data visualizations eventually.
"""
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
