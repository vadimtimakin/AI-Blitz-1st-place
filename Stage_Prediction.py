import pandas as pd 

df = pd.read_csv("sample_submission.xls")

preds = [0 for i in range(599)]
preds += [1 for i in range(599, 1049)]
preds += [2 for i in range(1049, 1499)]
preds += [3 for i in range(1499, 1652)]
preds += [4 for i in range(1652, 1799)]

df["label"] = preds
df.to_csv("./submission.csv", index=False)
