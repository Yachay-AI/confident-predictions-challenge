import pandas as pd
import numpy as np
import sys
import pickle
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import catboost as cb



def softmax(x):
    # Compute the exponential values for each element in the input array
    exps = np.exp(x - np.max(x))

    # Compute the softmax values by dividing the exponential of each element by the sum of exponentials
    return exps / np.sum(exps)


def count_chars(text):
    return len(text)


def count_words(text):
    return len(text.split())


def count_capital_words(text):
    return sum(map(str.isupper, text.split()))


def count_punctuations(text):
    punctuations = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    d = dict()
    for i in punctuations:
        d[str(i) + ' count'] = text.count(i)
    return d


def count_sent(text):
    return len(nltk.sent_tokenize(text))


def count_unique_words(text):
    return len(set(text.split()))


def count_htags(text):
    x = re.findall(r'(#w[A-Za-z0-9]*)', text)
    return len(x)


def count_mentions(text):
    x = re.findall(r'(@w[A-Za-z0-9]*)', text)
    return len(x)


def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)


def count_capital_chars(text):
    count = 0
    for i in text:
        if i.isupper():
            count += 1
    return count


# Load the data from a Parquet file into a pandas DataFrame.
data_frame = pd.read_parquet(sys.argv[1])

models = joblib.load("models/models.joblib")
model_cat = models['cat']
model_xgbr = models['xgb']
with open('models/class_acc.pkl', 'rb') as fp:
    acc_cl = pickle.load(fp)
with open('models/class_mean_dist.pkl', 'rb') as fp:
    avg_label_dist = pickle.load(fp)

TOP_CLASSES_COUNT = 10
# Initialize an empty list to store the maximum confidence values.
softmax_values_all = []
softmax_prob_label_all = []
softmax_mean_dist_label_all = []

# Iterate over the DataFrame rows.
for _, row in data_frame.iterrows():
    # Compute softmax for the 'raw_prediction' column of the current row.
    sofmax_row = softmax(row['raw_prediction'])

    softmax_values = np.sort(sofmax_row)[-TOP_CLASSES_COUNT:]
    softmax_values_all.append(softmax_values)

    softmax_arg = np.argsort(sofmax_row)[-TOP_CLASSES_COUNT:]
    softmax_arg_prob = [acc_cl[el] for el in softmax_arg]
    softmax_prob_label_all.append(softmax_arg_prob)

    softmax_mean_dist = []
    for el in softmax_arg:
        if el in avg_label_dist:
            softmax_mean_dist.append(avg_label_dist[el])
        else:
            softmax_mean_dist.append(0)

    softmax_mean_dist_label_all.append(softmax_mean_dist)

data_frame['pred'] = [x.argmax() for x in data_frame['raw_prediction']]
data_frame['acc_label'] = data_frame['pred'].apply(lambda x: acc_cl[x])
data_frame['mean_label_dist'] = data_frame['pred'].apply(lambda x: avg_label_dist[x])
# Add a new column 'confidence' to the DataFrame using the list of maximum confidence values.
data_frame['softmax'] = softmax_values_all
columns = ['cl_{}'.format(x + 1) for x in range(TOP_CLASSES_COUNT)]
data_frame[columns] = data_frame['softmax'].copy().tolist()

data_frame['mean_dist_top'] = softmax_mean_dist_label_all
data_frame['mean_acc_label_top'] = softmax_prob_label_all

columns = ['d_{}'.format(x + 1) for x in range(TOP_CLASSES_COUNT)]
data_frame[columns] = data_frame['mean_dist_top'].copy().tolist()

columns = ['a_{}'.format(x + 1) for x in range(TOP_CLASSES_COUNT)]
data_frame[columns] = data_frame['mean_acc_label_top'].copy().tolist()

data_frame['char_count'] = data_frame["text"].apply(lambda x: count_chars(x))
data_frame['word_count'] = data_frame["text"].apply(lambda x: count_words(x))
data_frame['sent_count'] = data_frame["text"].apply(lambda x: count_sent(x))
data_frame['capital_char_count'] = data_frame["text"].apply(lambda x: count_capital_chars(x))
data_frame['capital_word_count'] = data_frame["text"].apply(lambda x: count_capital_words(x))
data_frame['stopword_count'] = data_frame["text"].apply(lambda x: count_stopwords(x))
data_frame['unique_word_count'] = data_frame["text"].apply(lambda x: count_unique_words(x))
data_frame['htag_count'] = data_frame["text"].apply(lambda x: count_htags(x))
data_frame['mention_count'] = data_frame["text"].apply(lambda x: count_mentions(x))
data_frame['punct_count'] = data_frame["text"].apply(lambda x: count_punctuations(x))
data_frame['avg_wordlength'] = data_frame['char_count'] / data_frame['word_count']
data_frame['avg_sentlength'] = data_frame['word_count'] / data_frame['sent_count']
data_frame['unique_vs_words'] = data_frame['unique_word_count'] / data_frame['word_count']
data_frame['stopwords_vs_words'] = data_frame['stopword_count'] / data_frame['word_count']

X = data_frame.drop(
    ['text', 'raw_prediction', 'softmax', 'mean_acc_label_top', 'mean_dist_top', 'punct_count', 'confidence'], axis=1)
pred = model_cat.predict(X)
data_frame['score1'] = pred
pred = model_xgbr.predict(X)
data_frame['score2'] = pred
data_frame['score'] = 0.6 * data_frame['score1'] + 0.4 * data_frame['score2']

top_records_count = int(0.1 * len(data_frame))

# Sort the DataFrame by 'confidence' in descending order.
sorted_data_frame = data_frame.sort_values(by='score', ascending=True)

# Determine the number of top records to consider for computing mean distance.
top_records_count = int(0.1 * len(data_frame))

pd.Series(sorted_data_frame.iloc[:top_records_count].index).to_csv('submission.csv', index=False, header=None)

print(f'Submission saved to submission.csv, number of rows: {top_records_count}')
