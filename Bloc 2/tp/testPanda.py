import pandas as pd

s1 = pd.Series([1, 2, 3])

print(s1)

s2 = pd.Series([1, 2, 3], index=["A", "B", "C"])

print(s2)

s3 = pd.Series({"A": 1, "B": 2, "C": 3})

print(s3)