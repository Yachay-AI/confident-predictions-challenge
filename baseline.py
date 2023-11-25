import pandas as pd
import numpy as np
import sys
from joblib import load


def load_sklearn_model(model_path):
    return load(model_path)

data_frame = pd.read_parquet(sys.argv[1])

xgb_model = load_sklearn_model('xgb_model.joblib')
lgb_model = load_sklearn_model('lgb_model.joblib')

data_frame = data_frame[['text', 'raw_prediction', 'confidence']]
data_frame['text_length'] = data_frame['text'].apply(len)
data_frame['word_count'] = data_frame['text'].apply(lambda x: len(x.split()))
data_frame['hashtag_count'] = data_frame['text'].apply(lambda x: x.count('@'))
data_frame['link_count'] = data_frame['text'].apply(lambda x: x.count('https://'))
data_frame['exclamation_count'] = data_frame['text'].apply(lambda x: x.count('!'))
data_frame['question_count'] = data_frame['text'].apply(lambda x: x.count('?'))
data_frame = data_frame.join(data_frame['raw_prediction'].apply(pd.Series).add_prefix('raw_prediction_'))
data_frame.drop(['text'], axis=1, inplace=True)
rp = data_frame.raw_prediction
data_frame.drop(['raw_prediction'], axis=1, inplace=True)

# Add a new column 'confidence' to the DataFrame using the list of maximum confidence values.
data_frame['pred_distance'] = (lgb_model.predict(data_frame) + xgb_model.predict(data_frame)) / 2
data_frame['pred'] = [x.argmax() for x in rp]

# Sort the DataFrame by 'confidence' in descending order.
sorted_data_frame = data_frame.sort_values(by='pred_distance', ascending=True)

# Determine the number of top records to consider for computing mean distance.
top_records_count = int(0.1 * len(data_frame))

pd.Series(sorted_data_frame.iloc[:top_records_count].index).to_csv('submission.csv', index=False, header=None)

print(f'Submission saved to submission.csv, number of rows: {top_records_count}')
