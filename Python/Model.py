# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:28:45 2018

@author: t4nis
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:57:50 2018

@author: dchan
"""
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.externals.six import StringIO  
from IPython.display import Image 
import collections
from sklearn.tree import export_graphviz
from sklearn.model_selection import (GridSearchCV, train_test_split, StratifiedKFold)
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.metrics import classification_report


data = pd.read_csv("C:/Users/t4nis/Desktop/BIA 660/Project/Data.csv")

df1=pd.DataFrame(data)


print(df1)

#Removing $ in price 
price = df1.Price.str.replace('$',' ')
price = price.str.split(' ', expand = True, n = 3)
price.drop([0,2,3], axis = 1, inplace = True)
price.rename(columns = {1:"Price"}, inplace = True)
df1.drop('Price', axis = 1, inplace = True)
df1 = pd.concat([df1, price], axis = 1, sort =False)
df1['Price']=df1.Price.str.replace('+','')




#Retaining rating alone
Star = df1.Star.str.split(' ', expand = True, n = 1)
Star.drop(1, axis = 1, inplace = True)
df1.drop('Star', axis = 1, inplace = True)
df1 = pd.concat([df1, Star], axis = 1, sort = False)
df1.rename(columns = {0:"Star"}, inplace = True)

#Retaining number of reviews
df1.rename(columns = {"Number of Reviews":"Num_Rev"}, inplace = True)
Num_Rev = df1.Num_Rev.str.split(' ', expand = True, n = 1)
Num_Rev.drop(1, axis = 1, inplace = True)
df1.drop('Num_Rev', axis = 1, inplace = True)
df1 = pd.concat([df1, Num_Rev], axis = 1, sort = False)
df1.rename(columns = {0:"Num_Rev"}, inplace = True)

#Converting the columns
df1['Tag'] = (df1['Tag'] == "Amazon's Choice").astype(int).astype('category')
df1['Category'] = (df1['Category']).astype('category')
df1['Availability'] = (df1['Availability']).astype('category')
df1['Asin_number'] = (df1['Asin_number']).astype('category')
df1['Name'] = (df1['Name']).astype('category')
df1['Type of sale'] = (df1['Type of sale']).astype('category')
#df1['Num_rev'] = df1.Num_Rev.astype('float64')
#df1['Price'] = df1.Num_Rev.astype('float64')
#df1['Star'] = df1.Num_Rev.astype('float64')

#Removing blank rows
df1 = df1.dropna(subset = ['Name'])
df1['Availability'] = df1['Availability'].fillna("In Stock.")
df1['Type of sale'] = df1['Type of sale'].fillna("Ships from and sold by Amazon.com.")
df1 = df1.dropna(subset = ['Price'])
df1 = df1.dropna(subset = ['Star'])
df1 = df1.dropna(subset = ['Num_Rev'])
df1 = df1.dropna(subset = ['Category'])

#remove commas
df1['Price'] = df1.Price.str.replace(',','')
df1['Num_Rev'] = df1.Num_Rev.str.replace(',','')
#conversion datatype
df1['Num_Rev'] = df1.Num_Rev.astype('int64')
df1['Star'] = df1.Star.astype('float64')
df1['Price'] = df1.Price.astype('float64')
#df1.drop('Unnamed: 0', axis = 1, inplace = True)

#count of postive values
#len(df1[df1['Tag'] == 1])

column_label = ["Category","Availability","Type of sale"]
lb_make = LabelEncoder()
for column in column_label:
    df1[column] = lb_make.fit_transform(df1[column])
df1 = df1.set_index('Asin_number')
df1.drop('Name', axis = 1, inplace = True)

 
#a = pd.get_dummies(df1).head(1)


#Split dataset
y=df1[['Tag']]
x = df1.drop('Tag', axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=123)

#Sample
#SMOTE analysis
smt=SMOTE(random_state=123, ratio=1.0)
x_train_sam, y_train_sam=smt.fit_sample(x_train, y_train)
print(collections.Counter(y_train_sam))

#LR
sampled_logistic = LogisticRegression().fit(x_train_sam, y_train_sam)
sampled_pred = sampled_logistic.predict(x_test)
print(accuracy_score(y_test, sampled_pred))
print(confusion_matrix(y_test, sampled_pred))
recall_score(y_test, sampled_pred)

#GridSearch LR
pipe_lr = imbPipeline([('oversample', SMOTE(random_state=123,ratio=1)),
                       ('lr', LogisticRegression())])

skf = StratifiedKFold(n_splits=5) 
param_grid_lr =[ {'lr__C':[0.5,1,1.5,2],'lr__penalty':['l1','l2']}]
grid_lr = GridSearchCV(pipe_lr, param_grid_lr, return_train_score=True,
                    n_jobs=-1, scoring='roc_auc', cv=skf)
grid_lr.fit(x_train, y_train)
print(grid_lr.best_score_)

y_pred=grid_lr.predict(x_test)
print(classification_report(y_test,y_pred))
optimised_lr=grid_lr.best_estimator_
optimised_LR=optimised_lr.fit(x_train,y_train)
pred_lr_opt=optimised_LR.predict(x_test)
print(accuracy_score(y_test,pred_lr_opt))
print(confusion_matrix(y_test,pred_lr_opt))

#KNN
sampled_KNN = KNeighborsClassifier(n_neighbors=10).fit(x_train_sam, y_train_sam)
sampled_pred_KNN = sampled_KNN.predict(x_test)
print(accuracy_score(y_test, sampled_pred_KNN))
print(confusion_matrix(y_test, sampled_pred_KNN))
print(recall_score(y_test, sampled_pred_KNN))

#GridSearch KNN
pipe_kn = imbPipeline([('oversample', SMOTE(random_state=123,ratio=1)),
                       ('knn', KNeighborsClassifier())])

skf = StratifiedKFold(n_splits=5) 
param_grid_kn =[{'knn__n_neighbors': [1,3,5,7,9,11,13,15,17], 'knn__weights':['uniform','distance']}]
grid_kn = GridSearchCV(pipe_kn, param_grid_kn, return_train_score=True,
                    n_jobs=-1, scoring='roc_auc', cv=skf)
grid_kn.fit(x_train, y_train)
print(grid_kn.best_score_)

y_pred=grid_kn.predict(x_test)
print(classification_report(y_test,y_pred))
optimised_kn=grid_kn.best_estimator_
optimised_KN=optimised_kn.fit(x_train,y_train)
pred_kn_opt=optimised_KN.predict(x_test)
print(accuracy_score(y_test,pred_kn_opt))
print(confusion_matrix(y_test,pred_kn_opt))

#DT
sampled_DT = DecisionTreeClassifier().fit(x_train_sam, y_train_sam)
sampled_pred_DT = sampled_DT.predict(x_test)
print(accuracy_score(y_test, sampled_pred_DT))
print(confusion_matrix(y_test, sampled_pred_DT))
print(recall_score(y_test, sampled_pred_DT))

#GridSearch Decision Tree
pipe_dt = imbPipeline([('oversample', SMOTE(random_state=123,ratio=1)),
                       ('dt', DecisionTreeClassifier())])

skf = StratifiedKFold(n_splits=5) 
param_grid_dt =[{'dt__max_depth': [3,4,5,6,7,8,9,10,11,12],'dt__criterion':['gini','entropy']}]
grid_dt = GridSearchCV(pipe_dt, param_grid_dt, return_train_score=True,
                    n_jobs=-1, scoring='roc_auc', cv=skf)
grid_dt.fit(x_train, y_train)
print(grid_dt.best_score_)

y_pred=grid_dt.predict(x_test)
print(classification_report(y_test,y_pred))
optimised_dt=grid_dt.best_estimator_
optimised_DT=optimised_dt.fit(x_train,y_train)
pred_dt_opt=optimised_DT.predict(x_test)
print(accuracy_score(y_test,pred_dt_opt))
print(confusion_matrix(y_test,pred_dt_opt))

#RF
rf = RandomForestClassifier()
sampled_RF = rf.fit(x_train_sam, y_train_sam)
sampled_pred_RF = sampled_RF.predict(x_test)
print(accuracy_score(y_test, sampled_pred_RF))
print(confusion_matrix(y_test, sampled_pred_RF))
print(recall_score(y_test, sampled_pred_RF))

#Gridsearch for Random forest
pipe_rf = imbPipeline([
    ('oversample', SMOTE(random_state=123,ratio=1)),
    ('clf', RandomForestClassifier(random_state=123, n_jobs=-1))
    ])

skf = StratifiedKFold(n_splits=15) 
param_grid_rf = {'clf__max_depth': [1, 100],
             'clf__max_features': ['sqrt']}
grid_rf = GridSearchCV(pipe_rf, param_grid_rf, return_train_score=True,
                    n_jobs=-1, scoring='roc_auc', cv=skf)
grid_rf.fit(x_train, y_train)
y_pred=grid_rf.predict(x_test)
print(classification_report(y_test,y_pred))
optimised_rf=grid_rf.best_estimator_
optimised_randomforest=optimised_rf.fit(x_train,y_train)
pred_rf=optimised_randomforest.predict(x_test)
print(accuracy_score(y_test,pred_rf))
print(confusion_matrix(y_test,pred_rf))






















"""
df1['Star'] = df1.Star.astype('float64')
df1['Num_Rev'] = df1['Num_Rev'].str.replace(',','')
df1['Num_Rev'] = df1['Num_Rev'].fillna(0)
df1['Num_Rev'] = df1.Num_Rev.astype('int64')
df1['Price'] = df1['Price'].str.replace(',','')
df1['Price'] = df1['Price'].fillna(0)
df1['Price'] = df1.Price.astype('float64')
df1['Asin_number'] = df1.Asin_number.astype('str')
df1['Tag'].value_counts()
"""

