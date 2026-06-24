import pandas as pd
df = pd.read_parquet(rC:\Users\karlk\Repos\edSmith\data\hf_cache\chillies___ielts-writing-task-2-evaluation\default\0.0.0\ce4ab2f4e652ce08582f5488b23e1ebffa222e26\ielts-writing-task-2-evaluation-train.arrow)
print(df.columns.tolist())
print(df)#.head(8))
#print(df.iloc[0]['question'])
#print(df.iloc[0]['essay'])
#print(df.iloc[0]['feedback_text'])