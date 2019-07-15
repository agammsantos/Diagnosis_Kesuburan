import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


df=pd.read_csv('fertility.csv')
# print(df.head())

from sklearn.preprocessing import LabelEncoder
label1=LabelEncoder()
label2=LabelEncoder()
label3=LabelEncoder()
label4=LabelEncoder()
label5=LabelEncoder()
label6=LabelEncoder()
label7=LabelEncoder()
label8=LabelEncoder()
df.pop('Season')
df['Childish diseases']=label1.fit_transform(df['Childish diseases'])
df['Accident or serious trauma']=label2.fit_transform(df['Accident or serious trauma'])
df['Surgical intervention']=label3.fit_transform(df['Surgical intervention'])
df['High fevers in the last year']=label4.fit_transform(df['High fevers in the last year'])
df['Frequency of alcohol consumption']=label5.fit_transform(df['Frequency of alcohol consumption'])
df['Smoking habit']=label6.fit_transform(df['Smoking habit'])
# df.drop(columns=['Season','Childish diseases','Accident or serious trauma','Surgical intervention','High fevers in the last year','Frequency of alcohol consumption','Smoking habit'])
dfTarget=df.pop('Diagnosis')
dfTarget=label7.fit_transform(dfTarget)
# print(dfTarget)
# print(label1.classes_)
# print(label2.classes_)
# print(label3.classes_)
# print(label4.classes_)
# print(label5.classes_)
# print(label6.classes_)
# print(label7.classes_)
# print(dfTarget)
# print(df.iloc[1])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

coltrans=ColumnTransformer(
    [('OHE',OneHotEncoder(categories='auto'),[4,5,6])],
    remainder='passthrough'
)
dfohe=coltrans.fit_transform(df)
# dfohe5=coltrans5.fit_transform(df)
# dfohe6=coltrans6.fit_transform(df)
# dfohe7=coltrans7.fit_transform(df)
# print(df.columns.values)
# print(dfohe[1])
# print(dfohe5)
# print(dfohe6)
# print(dfohe7)

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(dfohe,dfTarget,test_size=0,random_state=12)

# logistic
modelLog=LogisticRegression(solver='liblinear',multi_class='auto')
modelLog.fit(xtr,ytr)

# dec tree
modelTree=DecisionTreeClassifier()
modelTree.fit(xtr,ytr)

# random forest
modelRandom=RandomForestClassifier() # n_estimators=50 : pembagian sub-sample untuk decision tree=50
modelRandom.fit(xtr,ytr)

# kmeans
# modelKmeans=KMeans(n_clusters=len(label7.classes_))
# modelKmeans.fit(xtr,ytr)

# extra tree
# modelExtra=ExtraTreesClassifier()
# modelExtra.fit(xtr,ytr)

# print(dfTarget)
# print(modelLog.predict(dfohe))
# print(modelTree.predict(dfohe))
# print(modelRandom.predict(dfohe))

# Fever ['less than 3 months ago' 'more than 3 months ago' 'no']
# Alcohol ['every day' 'hardly ever or never' 'once a week' 'several times a day' 'several times a week']
# Smoking ['daily' 'never' 'occasional']
# age
# Disease [0 1] = ['no' 'yes']
# Accident [0 1] = ['no' 'yes']
# Surgical [0 1] = ['no' 'yes']
# hours sit
# target: ['Altered' 'Normal']

def target(x):
    if x[0]==0:
        return label7.classes_[x[0]]
    elif x[0]==1:
        return label7.classes_[x[0]]

print('Arin, prediksi kesuburan:',target(modelLog.predict([[0,0,1,1,0,0,0,0,1,0,0,29,0,0,0,5]])),'(Logistic Regression)')
print('Arin, prediksi kesuburan:',target(modelTree.predict([[0,0,1,1,0,0,0,0,1,0,0,29,0,0,0,5]])),'(Decision Tree Classifier)')
print('Arin, prediksi kesuburan:',target(modelRandom.predict([[0,0,1,1,0,0,0,0,1,0,0,29,0,0,0,5]])),'(Random Forest Classifier)')
print(' ')
print('Bebi, prediksi kesuburan:',target(modelLog.predict([[0,0,1,0,0,0,0,1,0,1,0,31,0,1,1,16]])),'(Logistic Regression)')
print('Bebi, prediksi kesuburan:',target(modelTree.predict([[0,0,1,0,0,0,0,1,0,1,0,31,0,1,1,16]])),'(Decision Tree Classifier)')
print('Bebi, prediksi kesuburan:',target(modelRandom.predict([[0,0,1,0,0,0,0,1,0,1,0,31,0,1,1,16]])),'(Random Forest Classifier)')
print(' ')
print('Caca, prediksi kesuburan:',target(modelLog.predict([[1,0,0,0,1,0,0,0,0,1,0,25,1,0,0,7]])),'(Logistic Regression)')
print('Caca, prediksi kesuburan:',target(modelTree.predict([[1,0,0,0,1,0,0,0,0,1,0,25,1,0,0,7]])),'(Decision Tree Classifier)')
print('Caca, prediksi kesuburan:',target(modelRandom.predict([[1,0,0,0,1,0,0,0,0,1,0,25,1,0,0,7]])),'(Random Forest Classifier)')
print(' ')
print('Dini, prediksi kesuburan:',target(modelLog.predict([[0,0,1,0,1,0,0,0,1,0,0,28,0,1,1,16]])),'(Logistic Regression)')
print('Dini, prediksi kesuburan:',target(modelTree.predict([[0,0,1,0,1,0,0,0,1,0,0,28,0,1,1,16]])),'(Decision Tree Classifier)')
print('Dini, prediksi kesuburan:',target(modelRandom.predict([[0,0,1,0,1,0,0,0,1,0,0,28,0,1,1,16]])),'(Random Forest Classifier)')
print(' ')
print('Enno, prediksi kesuburan:',target(modelLog.predict([[0,0,1,0,1,0,0,0,0,1,0,42,1,0,0,8]])),'(Logistic Regression)')
print('Enno, prediksi kesuburan:',target(modelTree.predict([[0,0,1,0,1,0,0,0,0,1,0,42,1,0,0,8]])),'(Decision Tree Classifier)')
print('Enno, prediksi kesuburan:',target(modelRandom.predict([[0,0,1,0,1,0,0,0,0,1,0,42,1,0,0,8]])),'(Random Forest Classifier)')
