# Confident Predictions Selection: A Machine Learning Challenge
Welcome to the Confident Predictions Selection Machine Learning Challenge! The goal of this challenge is to construct a model or an algorithm that excels in selecting the top 10% of predictions with the lowest mean distance (error), derived from a previously trained machine learning model. Participants have the option to utilize either or both of two types of data provided: text data and raw model prediction outputs.

## Validation metrics 
- Mean Distance 
- CRI

## Challange Task
Your challenge is to design a model capable of selecting the top 10% of predictions that will exhibit the smallest mean distance. You have the option to work with either or both of the text data and raw predictions. Additionally, you may perform any form of aggregation or transformation on the raw predictions as you see fit.
The primary metric for this challenge is the mean distance of the selected top 10% predictions. Your objective is to minimize this value. As a secondary metric, the Class Representation Index (CRI) has been designed to evaluate how well the class distribution is preserved after a filtering operation is applied to a multi-class dataset, such as the selection of the most confident predictions in a classification model. In essence, CRI compares the class distribution before and after the filtering operation, giving a higher weight to classes that were initially larger. The primary purpose of this metric is to detect cases where a class is significantly less represented after filtering compared to its original size.

## Dataset 
[**Link to GDrive with dataset.**](https://drive.google.com/drive/folders/1KrB9mhCk5aOUOeltRZNl7Ph2VSoRDPZ3)
The dataset is sizable, with a total of 615,000 instances.

The dataset is composed of four primary components:

- `Text` This is a string data type, representing the content of tweets.
- `Distance` This is a float data type, acting as an error value for the machine learning predictions. The lower the distance, the more accurate the prediction.
- `True Label` This is an integer data type, ranging from 0 to 2999, indicating the true class value for the sample.
- `Raw Predictions` These are float values in a 3000x1 format, representing the logits from a 3000-class classification deep learning model built on a transformer architecture.

### Data
The data will be divided into training, development, and test datasets, with an 80/10/10 distribution, respectively. Full access will be provided to the training and development datasets, while the test set will be made available without the true distance and true label for evaluation purposes.

## Baseline Solution
The baseline solution for this challenge involves the calculation of the model's confidence as the maximum probability, which is derived from the softmax function applied to the raw prediction logits.
The baseline's validation scores are as follows:

```
Mean Distance = 738.15
CRI = 0.584
```

## Submission instructions
1. **Fork the Repository**: Start by forking the repository.

1. **Edit the Baseline Algorithm**: Make your improvements to the \`**baseline.py**\` file.

1. **Test on the Validation Set**: You can evaluate your algorithm's performance on the validation set using the following commands:
- To produce the \`**submission.csv**\` file: `python baseline.py relevance_challenge_valid.parquet`
- To display the scores: `python test.py relevance_challenge_valid.parquet`

4. **Run Predictions on the Test Set**: Once satisfied with your algorithm's performance on the validation set, run predictions for the test set:
`python baseline.py relevance_challenge_test.parquet`

This will produce the \`**submission.csv**\` file. Ensure the file name remains as \`**submission.csv**\` as it's crucial for the evaluation process.

5. **Create a Pull Request**: Submit a pull request with your modified code and the \`**submission.csv**\` file. An automated script will evaluate your submission and post the results as a comment on your pull request.


**&#9432;** **Note**: While the use of both textual features and raw predictions can potentially lead to better-performing models, this is not a mandatory requirement. Participants are encouraged to explore different strategies, which may include focusing solely on the text data, the raw predictions, or a combination of both.
<p align="center"> 👾</p>
We look forward to your innovative solutions in tackling this unique challenge! Further information about the evaluation process, the timeline for the challenge, and the specific submission guidelines will be provided soon. Stay tuned!

<p align="center"> <a href="https://discord.gg/msWFtcfmwe"><img src="https://cdn-icons-png.flaticon.com/512/3670/3670157.png" width=5% height=5%></img></a>     <a href="https://twitter.com/YachayAi"><img src="https://cdn-icons-png.flaticon.com/128/3670/3670151.png" width=5% height=5%></img></a>     <a href="https://www.reddit.com/user/yachay_ai"><img src="https://cdn-icons-png.flaticon.com/512/3670/3670226.png" width=5% height=5%></img></a></p>
