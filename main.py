import pandas as pd

df1 = pd.read_csv('data/c1.csv')
df2 = pd.read_csv('data/c2.csv')

print(len(df1))
print(len(df2))

print(df2.isnull())
