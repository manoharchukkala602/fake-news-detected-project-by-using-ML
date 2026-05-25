from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
true =pd.read_csv("True.csv")
fake =pd.read_csv("Fake.csv")
print(fake.head())
print(true.head())
fake["label"]=0
true["label"] =1
data =pd.concat([fake,true])
data =data.sample(frac=1)
data.reset_index(drop =True,inplace=True)
x=data["text"].astype(str)
y=data["label"]
count =TfidfVectorizer(stop_words="english")
x=count.fit_transform(x)
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)
model =LogisticRegression()
model.fit(x_train,y_train)
predict =model.predict(x_test)
accuracy =accuracy_score(y_test,predict)
print(f"total accuracy_score:{accuracy}")
news =input("Enter your news:")
vector =count.transform([news])
pre =model.predict(vector)
if pre[0]==1:
     print("news is true")
else:
    print("news is fake")

