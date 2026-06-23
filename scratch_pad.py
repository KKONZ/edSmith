import pandas as pd
df = pd.read_parquet(r"C:\Users\karlk\.claude\plugins\cache\edsmith\edsmith\0.1.0\edsmith_drive\sessions\ac63a699-9794-4885-b218-46f46bf31b37\feedback_iter0.parquet")
print(df.columns.tolist())
print(df.head(8))
print(df.iloc[0]['question'])
print(df.iloc[0]['essay'])
print(df.iloc[0]['feedback_text'])