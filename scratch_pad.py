import pandas as pd
df = pd.read_parquet(r"C:\Users\karlk\Repos\edSmith\edsmith_drive\sessions\03eb2c74-6d44-44e0-a5ba-ef85c6d6f3c2\feedback_iter1.parquet")
print(df.columns.tolist())
print(df.head(8))
print(df.iloc[0]['feedback_text'])

print(len(df))
print(df['score'].value_counts())
print(df['band'].value_counts())
#print("TASK RESPONSE")
#print(df.iloc[0]['question'])
#print(df.iloc[0]['essay'])
#print(df.iloc[0]['feedback_text'])


#print("Coherence")
#print(df.iloc[1]['question'])
#print(df.iloc[1]['essay'])
#print(df.iloc[1]['feedback_text'])

#print("GRAMMAR")
#print(df.iloc[3]['question'])
#print(df.iloc[3]['essay'])
#print(df.iloc[3]['feedback_text'])