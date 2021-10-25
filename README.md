# Twitter Sentiment Analysis
Sentiment Analysis is one of the application of Natural Language processing (NLP). It involves classifying a piece of text as negative,positive or neutral.

The objective of this project is to recognize whether the given tweet is oriented as positive(1), negative(-1) or neutral(0)

## Step 1: Loading the relevant libraries

## Step 2: Loading the data

## Step 3: Data Inspection
1. Text is a highly unstructured data with various types of noises present in it and the data cannot be analyzed properly with preprocessing. Thus data preprocessing involves first inspecting the data and then cleaning it for applying the model.

## Step 4: Data Cleaning
Cleaning raw data is an important step as it helps in getting rid of unwanted words and characters which helps in obtaining better features
It is always better to remove special symbols such as punctuation, special characters, numbers or terms which don't carry much weightage in the analysis

## Step 5: Text Normalization
Next step involves Text Normalization which involves extracting base terms from the morphological words. Before that we need to tokenize the tweets. Tokenization is the process of splitting a string of texts into tokens

## Step 6: Making Word Clouds (Visualization from tweets)
Visualizing the data is an important step for story telling and gaining insights
A WordCloud is a type of visualization where most frequent words appear in larger size and less frequent words in smaller size which helps in understanding the most common words used in the tweets.

## Step 7: Data Analyses
To analyze the preprocessed data, it needs to be converted into features. Depending upon the usage, a text data can be constructed using assorted techniques such as Bag of words, TF-IDF and Word Embeddings
Here we are using TF-IDF Features- TF-IDF means Term Frequency - Inverse Document Frequency. ... TF-IDF is better than Count Vectorizers because it not only focuses on the frequency of words present in the corpus but also provides the importance of the words.

## Step 8: Modeling
We will now build the model on the datasets with different feature sets prepared in the earlier sections
Here we will be using the LogisticRegression algorithm for the model
F1 score -- is used as the evaluation metric. It is the weighted average of Precision and Recall . Thus, this score takes both false positives and false negatives into account.
Logistic Regression -- It is a special case of linear regression when the outcome variable is categorical, where we are using log of odds as the dependent variables..It also predicts the probability of occurence of an event by fitting data to a logit function

## Step 9: Making confusion matrix
Now, it's time to wrap-up things. For this we are making confusion matrix which gives an idea about true positives, true negatives, false positives and false negatives in our final score¶
