import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import process_tweet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

df = pd.read_table('reviews.tsv')

small_df = df[:8000]
small_df = small_df.drop(small_df[(small_df["rating"] == 0.0)].index)
small_df.dropna(inplace=True)
small_df["clean_text"] = small_df["review_text"].apply(process_tweet)
print(small_df)

vectorizer = CountVectorizer()
corpus = small_df['clean_text']

X = vectorizer.fit_transform(corpus)
y = small_df["rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))