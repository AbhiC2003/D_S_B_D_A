import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
df=pd.read_csv("weatherAUS.csv")
print(df)
print(df.columns)
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
df=df.dropna()
print(df.isnull().sum())
df['RainTomorrow'].unique()
Y=df.RainTomorrow
print(Y.head())
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Rain Tomorrow']=label_encoder.fit_transform(df['RainTomorrow'])
print(df['Rain Tomorrow'].unique())
label_encoder=preprocessing.LabelEncoder()
df['WindGustDir']=label_encoder.fit_transform(df['WindGustDir'])
print(df['WindGustDir'].unique())
label_encoder = preprocessing.LabelEncoder()
df['RainToday']=label_encoder.fit_transform(df['RainToday'])
print(df['RainToday'].unique())
hm=sn.heatmap(data=df.corr(numeric_only = True),annot=True,annot_kws={'size':8})
sn.set(rc={'figure.figsize':(12,12)})
plt.show()
X=df.drop(['RainTomorrow', 'WindDir9am', 'WindDir3pm','WindSpeed9am'], axis='columns')
print(X.head())
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=10) 
from sklearn import svm
clf=svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
model=svm.SVC(kernel='poly')
model.fit(X_train, y_train)
y_pred-model.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))