import pandas as pd

df = pd.read_table('reviews.tsv')
print(df)

df_new = df.drop(df[(df["rating"] == 0.0)].index)
print(df_new)

df_positive_1 = df_new.drop(df_new[(df_new["rating"] == 1.0)].index)
df_positive_2 = df_positive_1.drop(df_positive_1[(df_positive_1["rating"] == 2.0)].index)
df_final_positive = df_positive_2.drop(df_positive_2[(df_positive_2["rating"] == 3.0)].index)
print(df_final_positive)

df_negative = df_new.drop(df_new[(df_new["rating"] == 4.0)].index)
df_negative_final = df_negative.drop(df_negative[(df_negative["rating"] == 5.0)].index)
print(df_negative_final)

