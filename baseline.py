import pandas as pd
import numpy as np
import sys
import catboost as cb
import pickle


def softmax(x):
    # Compute the exponential values for each element in the input array
    exps = np.exp(x - np.max(x))

    # Compute the softmax values by dividing the exponential of each element by the sum of exponentials
    return exps / np.sum(exps)


# Load the data from a Parquet file into a pandas DataFrame.
data_frame = pd.read_parquet(sys.argv[1])

# Load model and extra files
model = cb.CatBoostRegressor()
model.load_model('models/reg.pkl')
with open('models/class_acc.pkl', 'rb') as fp:
    class_acc = pickle.load(fp)
with open('models/class_mean_dist.pkl', 'rb') as fp:
    class_mean_dist = pickle.load(fp)

TOP_CLASSES_COUNT = 6
# Initialize an empty list to store the maximum confidence values.
softmax_values_all = []

# Iterate over the DataFrame rows.
for _, row in data_frame.iterrows():
    # Compute softmax for the 'raw_prediction' column of the current row.
    sofmax_row = softmax(row['raw_prediction'])

    softmax_values = np.sort(sofmax_row)[-TOP_CLASSES_COUNT:]
    softmax_values_all.append(softmax_values)

data_frame['pred'] = [x.argmax() for x in data_frame['raw_prediction']]
data_frame['acc_label'] = data_frame['pred'].apply(lambda x: class_acc[x])
data_frame['mean_label_dist'] = data_frame['pred'].apply(lambda x: class_mean_dist[x])

# Add a new column 'confidence' to the DataFrame using the list of maximum confidence values.
data_frame['softmax'] = softmax_values_all
columns = ['cl_{}'.format(x + 1) for x in range(TOP_CLASSES_COUNT)]
data_frame[columns] = data_frame['softmax'].copy().tolist()

data_frame['pred'] = [x.argmax() for x in data_frame['raw_prediction']]

X = data_frame.drop(['text', 'raw_prediction', 'softmax', 'confidence'],
                    axis=1)
pred = model.predict(X)
data_frame['score'] = pred

# Sort the DataFrame by 'confidence' in descending order.
sorted_data_frame = data_frame.sort_values(by='score', ascending=True)

# Determine the number of top records to consider for computing mean distance.
top_records_count = int(0.1 * len(data_frame))

pd.Series(sorted_data_frame.iloc[:top_records_count].index).to_csv('submission.csv', index=False, header=None)

print(f'Submission saved to submission.csv, number of rows: {top_records_count}')
