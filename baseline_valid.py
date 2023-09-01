import pandas as pd
import numpy as np

# Class imbalance ratio
def CIR(y_before, y_after, verbose=False):
    expected_ratio = len(y_after) / len(y_before)
    unique_classes = np.unique(y_before)
    scores = []
    weights = []

    for c in unique_classes:
        count_before = np.sum(y_before == c)
        count_after = np.sum(y_after == c)

        score = min((count_after / count_before) / expected_ratio, 1)
        weight = count_before

        scores.append(score)
        weights.append(weight)
        
        if verbose:
            print(f"Class {c}: score = {score}, weight = {weight}")

    CIR = np.average(scores, weights=weights)
    return CIR



def softmax(x):
    # Compute the exponential values for each element in the input array
    exps = np.exp(x - np.max(x))

    # Compute the softmax values by dividing the exponential of each element by the sum of exponentials
    return exps / np.sum(exps)


# Load the data from a Parquet file into a pandas DataFrame.
data_frame = pd.read_parquet('relevance_challenge_valid.parquet')

# Initialize an empty list to store the maximum confidence values.
max_confidences = []

# Iterate over the DataFrame rows.
for _, row in data_frame.iterrows():
    # Compute softmax for the 'raw_prediction' column of the current row.
    softmax_values = softmax(row['raw_prediction'])
    
    # Find the maximum confidence value and append it to the list.
    max_confidences.append(softmax_values.max())

# Add a new column 'confidence' to the DataFrame using the list of maximum confidence values.
data_frame['confidence'] = max_confidences
data_frame['pred'] = [x.argmax() for x in data_frame['raw_prediction']]

# Sort the DataFrame by 'confidence' in descending order.
sorted_data_frame = data_frame.sort_values(by='confidence', ascending=False)

# Determine the number of top records to consider for computing mean distance.
top_records_count = int(0.1 * len(data_frame))

# Compute and print the mean of the 'distance' column for the top 10% records.
mean_distance = sorted_data_frame.iloc[:top_records_count]['distance'].mean()
print(mean_distance)

# Compute and print Class Imbalance Ratio
print(CIR(data_frame['label'], sorted_data_frame.iloc[:top_records_count]['label']))