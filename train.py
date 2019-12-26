import os
import shutil

import pandas as pd
from svc_model import SVCModel
from sklearn.model_selection import train_test_split


# X chứa data feature, y chứa target
train_data = []
with open("data.txt", encoding="utf8") as f:
    for i, line in enumerate(f):
        if i % 2 == 0:
            text = line.strip()
        else:
            label = line.strip()
            score = label.split("\t")[0]
            train_data.append({"feature": text, "target": score})
df_train = pd.DataFrame(train_data)

test_data = []
with open("test_data.txt", encoding="utf8") as f:
    for i, line in enumerate(f):
        if i % 2 == 0:
            text = line.strip()
        else:
            label = line.strip()
            score = label.split("\t")[0]
            test_data.append({"feature": text, "target": score})
df_test = pd.DataFrame(test_data)

#init model
model = SVCModel()
clf = model.clf.fit(df_train["feature"], df_train.target)

predicted = clf.predict(df_test["feature"])

print(predicted)
print(clf.predict_proba(df_test["feature"]))