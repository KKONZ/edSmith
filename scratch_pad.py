import pandas as pd
df = pd.read_parquet(r"C:\Users\karlk\Repos\edSmith\edsmith_drive\sessions\282d5efa-8508-47ba-b72d-1783471a9744\data\test.parquet")
print(df.columns.tolist())
#print(df)#.head(8))

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