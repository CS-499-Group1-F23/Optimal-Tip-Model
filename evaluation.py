import pandas as pd


# Function to calculate and print the percentage of good, bad, and zero tips from a predictions file.
def calculate_tip_percentages(predictions_file):
    # Read the predictions file into a DataFrame.
    df = pd.read_csv(predictions_file)

    # Define the threshold for considering a tip as good (12% of the total amount).
    threshold = 0.12  # (12% of Total_Amount_USD)

    # Calculate the tip percentage for each entry in the DataFrame.
    df['Tip_Percentage'] = df['Predicted_Tip_USD'] / df['Total_Amount_USD']

    # Calculate the percentage of entries where the tip is above the threshold (good tips).
    good_tip_percentage = (df[df['Tip_Percentage'] > threshold].shape[0] / df.shape[0]) * 100

    # Calculate the percentage of entries where the tip is positive but below or equal to the threshold (bad tips).
    bad_tip_percentage = (df[(df['Tip_Percentage'] <= threshold) & (df['Tip_Percentage'] > 0)].shape[0] / df.shape[
        0]) * 100

    # Calculate the percentage of entries with zero tip.
    zero_tip_percentage = (df[df['Tip_Percentage'] == 0].shape[0] / df.shape[0]) * 100

    # Print out the calculated percentages.
    print(f"Percentage of good tip data: {good_tip_percentage:.2f}%")
    print(f"Percentage of bad tip data: {bad_tip_percentage:.2f}%")
    print(f"Percentage of zero tip data: {zero_tip_percentage:.2f}%")


# Specify the file containing the predictions data.
predictions_file = 'lrpredictions.csv'

# Call the function with the specified file.
calculate_tip_percentages(predictions_file)
