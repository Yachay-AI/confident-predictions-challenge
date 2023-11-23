import os
import sys
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
    
if os.path.isfile('submission.csv'):
    # Load the data from a Parquet file into a pandas DataFrame.
    if sys.argv[1].find('.csv') != -1:
        data_frame = pd.read_csv(sys.argv[1], index_col=0)
    elif sys.argv[1].find('.parquet') != -1:
        data_frame = pd.read_parquet(sys.argv[1])
    else:
        print('Please specify test file as first argument, it must be in csv or parquet format')
        raise ValueError()

    print(data_frame.head())
    answer = pd.read_csv('submission.csv', header=None)[0]

    # Determine the number of top records to consider for computing mean distance.
    top_records_count = int(0.1 * len(data_frame))

    if len(answer) != top_records_count:
        print(f'Submission file must include {top_records_count} ids. Found {len(answer)} ids.')
        raise ValueError()
        
    sorted_data_frame = data_frame.loc[answer]


    # Compute and print the mean of the 'distance' column for the top 10% records.
    mean_distance = sorted_data_frame.iloc[:top_records_count]['distance'].mean()
    #print(mean_distance)
    sorted_df_absolute_min = data_frame.sort_values(by='distance', ascending=True).iloc[:len(answer)]

    # Compute and print Class Imbalance Ratio
    cir = CIR(data_frame['label'], sorted_data_frame.iloc[:top_records_count]['label'])

    print(f'Submission scores: Mean distance={mean_distance:.2f}, Class Imbalance Ratio={cir:.2f}')
    print(f"Absolute min distance={sorted_df_absolute_min['distance'].mean():.2f}, len={len(sorted_df_absolute_min)}")
else:
    print('Could not find submission.csv file in the root folder, skipping')