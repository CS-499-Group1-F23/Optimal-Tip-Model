import joblib
import pandas as pd

def get_tip(model, test_input):
    # Calculate predicted rack times for the test set
    # Find the threshold rack time using the median of predicted rack times
    threshold_rack_time = 10

    # Extract feature weights and intercept from the trained model
    feature_weights = model.coef_
    intercept = model.intercept_

    predicted_optimal_tips = (threshold_rack_time - (feature_weights[0]* test_input['total_amount_USD']) - (test_input['Area_sqmi']*feature_weights[2]) - intercept) / (feature_weights[1])
    predicted_optimal_tips = float(predicted_optimal_tips.iloc[0])
    return  round(predicted_optimal_tips, 2)

def main():
    # Load model
    model = joblib.load('lr_model_2023-11-30.pkl')

    for i in range(0, 3):
        # Define test input
        test_input = float(input("Enter total amount: "))
        test_input_data = {
            'total_amount_USD': [test_input],
            'Tip_USD': [0.0],
            'Area_sqmi': [8.477078322]
        }

        test_input = pd.DataFrame(test_input_data)

        # Get suggested tip and print result for test inputs
        suggested_tip = get_tip(model, test_input)
        print(f"Optimal tip: ${suggested_tip}")
    

if __name__ == "__main__":
    main()