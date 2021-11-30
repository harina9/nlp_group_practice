import pandas as pd
import numpy as np

from utils import *

df = pd.read_table('reviews.tsv')

df_new = df.drop(df[(df["rating"] == 0.0)].index)

df_positive_1 = df_new.drop(df_new[(df_new["rating"] == 1.0)].index)
df_positive_2 = df_positive_1.drop(df_positive_1[(df_positive_1["rating"] == 2.0)].index)
df_final_positive = df_positive_2.drop(df_positive_2[(df_positive_2["rating"] == 3.0)].index)
df_final_positive = df_final_positive.dropna(axis=0, how="any")
df_final_positive = df_final_positive.reset_index(drop=True)


df_negative = df_new.drop(df_new[(df_new["rating"] == 4.0)].index)
df_negative_final = df_negative.drop(df_negative[(df_negative["rating"] == 5.0)].index)
df_negative_final = df_negative_final.dropna(axis=0, how="any")
df_negative_final = df_negative_final.reset_index(drop=True)


test_pos = df_negative_final.iloc[4000:8000]
train_pos = df_negative_final.iloc[0:4000]
test_neg = df_final_positive.iloc[4000:8000]
train_neg = df_final_positive.iloc[:4000]


train_x = pd.concat([train_pos, train_neg], ignore_index=True)
test_x = pd.concat([test_neg, test_pos], ignore_index=True)

# print(train_x)

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# print(train_y)
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

review = train_x['review_text'].astype('str').tolist()

freqs = build_freqs(review, train_y)

# # check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

print('This is an example of a review: \n', train_x['review_text'].astype('str').iloc[0])
print('\nThis is an example of the processed version of the review: \n', process_tweet(train_x['review_text'].astype('str').iloc[0]))

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def sigmoid(z):
    """
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    """
    h = 1 / (1 + np.exp(-z))
    return h


# Testing your function
if sigmoid(0) == 0.5:
    print("SUCCESS!")
else:
    print("Oops!")

if sigmoid(4.92) == 0.9927537604041685:
    print("CORRECT!")
else:
    print("Oops again!")

# verify that when the model predicts close to 1, but the actual label is 0, the loss is a large positive value
-1 * (1 - 0) * np.log(1 - 0.9999)  # loss is about 9.2

# verify that when the model predicts close to 0 but the actual label is 1, the loss is a large positive value
-1 * np.log(0.0001)  # loss is about 9.2


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def gradientDescent(x, y, theta, alpha, num_iters):
    """
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # get 'm', the number of rows in matrix x
    m = len(x)

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = (
            -float(1)
            / m
            * (y.transpose() * np.log(h) + (1 - y).transpose() * np.log(1 - h))
        )

        # update the weights theta
        theta = theta - (alpha / m) * (np.mat(x.transpose()) * np.mat((h - y)))

    J = float(J)
    return J, theta


# Check the function
# Construct a synthetic test case using numpy PRNG functions
np.random.seed(1)
# X input is 10 x 3 with ones for the bias terms
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
# Y Labels are 10 x 1
tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

# Apply gradient descent
tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
print(f"The cost after training is {tmp_J:.8f}.")

# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def extract_features(tweet, freqs):
    """
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    """
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    # loop through each word in the list of words
    for word in word_l:
        # increment the word count for the positive label 1
        x[0, 1] += freqs.get((word, 1.0), 0)

        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0.0), 0)

    ### END CODE HERE ###
    assert x.shape == (1, 3)
    return x


# Check your function

# test 1
# test on training data
tmp1 = extract_features(review[0], freqs)
print(tmp1)

# test 2:
# check for when the words are not in the freqs dictionary
tmp2 = extract_features("blorb bleeeeb bloooob", freqs)
print(tmp2)

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(review), 3))
for i in range(len(review)):
    X[i, :] = extract_features(review[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")

# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def predict_tweet(tweet, freqs, theta):
    """
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(x * theta)

    ### END CODE HERE ###

    return y_pred


# Run this cell to test your function
for tweet in [
    "I am happy",
    "I am bad",
    "this movie should have been great.",
    "great",
    "great great",
    "great great great",
    "great great great great",
]:
    print("%s -> %f" % (tweet, predict_tweet(tweet, freqs, theta)))


# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def test_logistic_regression(review, test_y, freqs, theta):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """

    # the list for storing predictions
    y_hat = []

    for one in review:
        # get the label prediction for the tweet
        y_pred = predict_tweet(one, freqs, theta)

        if y_pred > 0.5:
            y_hat.append(1.0)
        else:
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator

    if np.asarray(y_hat).all() == np.squeeze(test_y).all():
        accuracy = (sum(np.asarray(y_hat)) + sum(np.squeeze(test_y))) / len(review)

    return accuracy


tmp_accuracy = test_logistic_regression(review, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

# Some error analysis done for you
print("Label Predicted Tweet")
for x, y in zip(review, test_y):
    y_hat = predict_tweet(x, freqs, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print("THE TWEET IS:", x)
        print("THE PROCESSED TWEET IS:", process_tweet(x))
        print(
            "%d\t%0.8f\t%s"
            % (y, y_hat, " ".join(process_tweet(x)).encode("ascii", "ignore"))
        )

# Feel free to change the tweet below
my_review = "Больше сюда не придем, отвратительное место"
print(process_tweet(my_review))
y_hat = predict_tweet(my_review, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")

