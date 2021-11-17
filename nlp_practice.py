import pandas as pd
import numpy as np

from utils import *

df = pd.read_table('reviews.tsv')
#print(df)

df_new = df.drop(df[(df["rating"] == 0.0)].index)
#print(df_new)

df_positive_1 = df_new.drop(df_new[(df_new["rating"] == 1.0)].index)
df_positive_2 = df_positive_1.drop(df_positive_1[(df_positive_1["rating"] == 2.0)].index)
df_final_positive = df_positive_2.drop(df_positive_2[(df_positive_2["rating"] == 3.0)].index)


df_negative = df_new.drop(df_new[(df_new["rating"] == 4.0)].index)
df_negative_final = df_negative.drop(df_negative[(df_negative["rating"] == 5.0)].index)


test_pos = df_negative_final.iloc[42677:]
train_pos = df_negative_final.iloc[0:42677]
test_neg = df_final_positive.iloc[220272:]
train_neg = df_final_positive.iloc[:220272]


train_x = pd.concat([train_pos, train_neg], axis=1)
test_x = pd.concat([test_neg, test_pos], axis=1)


train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

freqs = build_freqs(train_x, train_y)

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

#print('This is an example of a positive tweet: \n', train_x.iloc[0])
#train_x['review_text'].iloc[0],  dtype="string", expand=False
#print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x['review_text'].iloc[0]))

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

