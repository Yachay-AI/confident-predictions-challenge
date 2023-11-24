import pandas as pd
import numpy as np
import sys
from joblib import load
import scipy.stats

def softmax(x):
    # Compute the exponential values for each element in the input array
    exps = np.exp(x - np.max(x))

    # Compute the softmax values by dividing the exponential of each element by the sum of exponentials
    return exps / np.sum(exps)

def transform_array(arr, length):
    if len(arr) > length:
        return arr[:length]  # Truncate
    else:
        return np.pad(arr, (0, length - len(arr)), 'constant')  # Pad


# Load the data from a Parquet file into a pandas DataFrame.
data_frame = pd.read_parquet(sys.argv[1])

# Initialize an empty list to store the maximum confidence values.
max_confidences = []

# Iterate over the DataFrame rows.
for _, row in data_frame.iterrows():
    # Compute softmax for the 'raw_prediction' column of the current row.
    softmax_values = softmax(row['raw_prediction'])
    
    # Find the maximum confidence value and append it to the list.
    max_confidences.append(softmax_values.max())

model = load('xgboost_model.joblib')

# Add a new column 'confidence' to the DataFrame using the list of maximum confidence values.
data_frame['confidence'] = max_confidences
X = np.stack(data_frame['raw_prediction'])


# Basic statistical features
mean_confidence = np.mean(X, axis=1)
std_confidence = np.std(X, axis=1)
max_confidence = np.max(X, axis=1)
min_confidence = np.min(X, axis=1)
sum_confidence = np.sum(X, axis=1)
median_confidence = np.median(X, axis=1)

# Additional percentiles
percentile_25 = np.percentile(X, 25, axis=1)
percentile_75 = np.percentile(X, 75, axis=1)
percentile_10 = np.percentile(X, 10, axis=1)
percentile_90 = np.percentile(X, 90, axis=1)

# Indices (positions) of max, min, median
argmax_confidence = np.argmax(X, axis=1)
argmin_confidence = np.argmin(X, axis=1)
argmedian_confidence = np.argmin(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1)

# Skewness and Kurtosis
skew_confidence = np.apply_along_axis(lambda x: scipy.stats.skew(x), axis=1, arr=X)
kurtosis_confidence = np.apply_along_axis(lambda x: scipy.stats.kurtosis(x), axis=1, arr=X)

# Range (max - min)
range_confidence = max_confidence - min_confidence

# Mean Absolute Deviation (MAD)
mad_confidence = np.mean(np.abs(X - np.mean(X, axis=1, keepdims=True)), axis=1)



# Cumulative Sum and Product
cumulative_sum_confidence = np.cumsum(X, axis=1).mean(axis=1)

# Difference Between Consecutive Features and Moving Average
difference_confidence = np.diff(X, axis=1).mean(axis=1)


# Softmax operation
softmax_confidence = scipy.special.softmax(X, axis=1)

# Taking top 5 values after Softmax
top_5_softmax = np.sort(softmax_confidence, axis=1)[:, -5:]

# Features for each of the top 5 softmax values
top_1_softmax = top_5_softmax[:, -1]
top_2_softmax = top_5_softmax[:, -2]
top_3_softmax = top_5_softmax[:, -3]
top_4_softmax = top_5_softmax[:, -4]
top_5_softmax = top_5_softmax[:, -5]


# Combine all features into a single 2D array
new_features = np.column_stack(
    (mean_confidence, std_confidence, max_confidence, min_confidence, sum_confidence,
     median_confidence, percentile_25, percentile_75, percentile_10, percentile_90,
     argmax_confidence, argmin_confidence, argmedian_confidence, skew_confidence, kurtosis_confidence,
     range_confidence, mad_confidence, cumulative_sum_confidence, difference_confidence,
    top_1_softmax, top_2_softmax, top_3_softmax, top_4_softmax, top_5_softmax))


import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
import textstat
from gensim.models import Word2Vec
from collections import Counter

# Ensure the necessary NLTK data is downloaded
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_features(df, text_column):
    # Helper functions for different features
    def length_based_features(text):
        chars = len(text)
        words = len(text.split())
        avg_word_length = chars / words if words else 0
        return [chars, words, avg_word_length]

    def pos_features(text):
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        pos_counts = Counter(tag for word, tag in pos_tags)
        return list(pos_counts.values())

    def ner_features(text):
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        named_ents = nltk.ne_chunk(pos_tags, binary=True)
        return len([chunk for chunk in named_ents if hasattr(chunk, 'label') and chunk.label() == 'NE'])

    def sentiment_score(text):
        return TextBlob(text).sentiment.polarity

    def readability_scores(text):
        flesch_reading = textstat.flesch_reading_ease(text)
        gunning_fog = textstat.gunning_fog(text)
        return [flesch_reading, gunning_fog]

    # Initialize lists to store each feature
    lengths, sentiments, readabilities, ners = [], [], [], []

    # Iterate through each text entry and extract features
    for text in df[text_column]:
        lengths.append(length_based_features(text))
        sentiments.append(sentiment_score(text))
        readabilities.append(readability_scores(text))
        ners.append(ner_features(text))

    # Convert lists to NumPy arrays
    lengths = np.array(lengths)
    sentiments = np.array(sentiments).reshape(-1, 1)
    readabilities = np.array(readabilities)
    ners = np.array(ners).reshape(-1, 1)

    # Concatenate all features into a single array
    features = np.concatenate([lengths, sentiments, readabilities, ners], axis=1)

    return features


text_features = extract_features(data_frame, 'text')

new_features = np.column_stack((new_features, text_features))



data_frame['pred'] = model.predict(new_features)

#data_frame['pred'] = [x.argmax() for x in data_frame['raw_prediction']]

# Sort the DataFrame by 'confidence' in descending order.
sorted_data_frame = data_frame.sort_values(by='pred', ascending=True)

# Determine the number of top records to consider for computing mean distance.
top_records_count = int(0.1 * len(data_frame))

pd.Series(sorted_data_frame.iloc[:top_records_count].index).to_csv('submission.csv', index=False, header=None)

print(f'Submission saved to submission.csv, number of rows: {top_records_count}')