import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import sys

def softmax(x):
    # Compute the exponential values for each element in the input array
    exps = np.exp(x - np.max(x))

    # Compute the softmax values by dividing the exponential of each element by the sum of exponentials
    return exps / np.sum(exps)


# Load the data from a Parquet file into a pandas DataFrame.
data_frame = pd.read_parquet(sys.argv[1])

# Initialize an empty list to store the maximum confidence values.
max_confidences = []
probabilities = []
# Iterate over the DataFrame rows.
for _, row in data_frame.iterrows():
    # Compute softmax for the 'raw_prediction' column of the current row.
    softmax_values = softmax(row['raw_prediction'])
    probabilities.append(softmax_values)
    # Find the maximum confidence value and append it to the list.
    max_confidences.append(softmax_values.max())

# Add a new column 'confidence' to the DataFrame using the list of maximum confidence values.
data_frame['confidence'] = max_confidences
data_frame['pred'] = [x.argmax() for x in data_frame['raw_prediction']]
data_frame['probabilities'] = probabilities

bins = [0, 0.1, 0.2, 0.3, 1.0]
probabilities = data_frame['probabilities'].values

histograms = []
for row in probabilities:
    histograms.append(np.histogram(row, bins=bins)[0])
    
histograms = np.array(histograms)
histograms = np.delete(histograms, 0, 1)

column_names = [f'{bins[i]}_{bins[i+1]}' for i in range(len(bins) - 1)]
histograms = histograms.T

data_frame[column_names[1]] = histograms[0]
data_frame[column_names[2]] = histograms[1]
data_frame[column_names[3]] = histograms[2]

data_frame['peaks'] = data_frame['0.1_0.2'] + data_frame['0.2_0.3'] + data_frame['0.3_1.0']

data_frame['text'] = data_frame['text'].str.replace(r'@[\w]+','')
data_frame['text'] = data_frame['text'].str.replace(r'https?://\S+|www\.\S+','')

data_frame['text_length'] = data_frame['text'].str.len()

with open('trained_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)
data_frame['score'] = trained_model.predict(data_frame[['confidence', 'peaks','text_length']])

sorted_data_frame = data_frame.sort_values(by=['score'], ascending=False)
# Determine the number of top records to consider for computing mean distance.
top_records_count = int(0.1 * len(data_frame))

pd.Series(sorted_data_frame.iloc[:top_records_count].index).to_csv('submission.csv', index=False, header=None)

print(f'Submission saved to submission.csv, number of rows: {top_records_count}')