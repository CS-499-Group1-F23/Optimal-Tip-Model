import pandas as pd

def calculate_tip_percentages(predictions_file):
    df = pd.read_csv(predictions_file)

    threshold = 0.12  # (12% of Total_Amount_USD)
    df['Tip_Percentage'] = df['Predicted_Tip_USD'] / df['Total_Amount_USD']
    
    good_tip_percentage = (df[df['Tip_Percentage'] > threshold].shape[0] / df.shape[0]) * 100
    bad_tip_percentage = (df[(df['Tip_Percentage'] <= threshold) & (df['Tip_Percentage'] > 0)].shape[0] / df.shape[0]) * 100
    zero_tip_percentage = (df[df['Tip_Percentage'] == 0].shape[0] / df.shape[0]) * 100

    print(f"Percentage of good tip data: {good_tip_percentage:.2f}%")
    print(f"Percentage of bad tip data: {bad_tip_percentage:.2f}%")
    print(f"Percentage of zero tip data: {zero_tip_percentage:.2f}%")

predictions_file = 'lrpredictions.csv'  
calculate_tip_percentages(predictions_file)
