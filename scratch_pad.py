import pandas as pd
df = pd.read_parquet(r"C:\Users\karlk\Repos\edSmith\edsmith_drive\sessions\22718419-2f92-455a-b398-608b8ddfffa8\feedback_iter0.parquet")
print(df.columns.tolist())
print(df.head(8))
print(df.iloc[0]['feedback_text'])

print(len(df))

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