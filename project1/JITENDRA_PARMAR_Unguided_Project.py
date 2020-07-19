import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import time

import warnings
warnings.filterwarnings(action="ignore")

print("CODE PROPOSED BY : JITENDRA PARMAR")
print("Email Id : jitendra.parmar_cs18@gla.ac.in")



pd.set_option('display.width', 500)
pd.set_option('display.max_column', 20)

data = pd.read_csv('HeartAttack_data Unguided_Project.csv')


data.replace('?', np.nan , inplace=True)
from sklearn.impute import SimpleImputer
data = data.fillna(data.median())

print(data.isnull().sum())
column=['trestbps','chol','fbs','restecg','thalach','exang']
data[column]=data[column].fillna(data.mode().iloc[0])
data['age'].fillna(data['age'].mean(),inplace=True)
data['sex'].fillna(data['sex'].mean(),inplace=True)
data['cp'].fillna(data['cp'].mean(),inplace=True)
data['oldpeak'].fillna(data['oldpeak'].mean(),inplace=True)
print(data)





data.loc[:,'trestbps':'exang']=data.loc[:,'trestbps':'exang'].applymap(float)

print("\n\n Heart attack data:- \n\n",data.head(5))

print("shape of heart attack data :- \n\n",data.shape)

print("\n\n\nHeart Attack data decription : \n")
print( data.describe() )

print( "\n\n\ndata.num.unique() : " , data.num.unique() )

data.rename(columns={'num':'target'} , inplace=True)

print(data.columns)

plt.hist(data['target'])
plt.title('taget (1=Yes , 0=No)')
plt.show()

print("checking null value:\n",data.isnull())

print("checking the nan value: \n\n" , data.info())


print("\n\n\ndata.groupby('target').size()\n")
print(data.groupby('target').size())

data.plot(kind='density', subplots=True, layout=(4,3), sharex=False)
plt.show()



names=['age','sex', ' cp',' trestbps','chol','fbs' ,  'restecg' ,'thalach' ,'exang',  'oldpeak' , 'num']
fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(data.corr() )
ax1.grid(True)
plt.title('Heart Attack Attributes Correlation')
ax1.set_xticklabels(names)
ax1.set_yticklabels(names)
# Add colorbar, make sure to specify tick locations to match desired ticklabels
fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
plt.show()

Y = data['target'].values
X = data.drop('target', axis=1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.33, random_state=7)


print("shape of x_train data :\n\n",X_train.shape)

print("shape of x_test data :\n\n",X_test.shape)

#These following algorithm we will check (spot check)

#1) Classification and Regression Trees (CART),
#2) Support Vector Machines (SVM),
#3) Gaussian Naive Bayes (NB)
#4) k-Nearest Neighbors (KNN).


models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))



num_folds = 10

results = []
names = []

for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=7)
    startTime = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    endTime = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), endTime-startTime))




fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()




# Standardize the dataset
pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))


results = []
names = []



print("\n\n\nAccuracies of algorithm after scaled dataset\n")



with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=7)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))



#Performance Comparison after Scaled Data

#*******************************************************************

fig = plt.figure()
fig.suptitle('Performance Comparison after Scaled Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
model = SVC()
start = time.time()
model.fit(X_train_scaled, Y_train)   #Training of algorithm
end = time.time()
print( "\n\nSVM Training Completed. It's Run Time: %f" % (end-start))


# estimate accuracy on test dataset
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print("All predictions done successfully by SVM Machine Learning Algorithms")
print("\n\nAccuracy score %f" % accuracy_score(Y_test, predictions))




print("\n\n")
print("confusion_matrix = \n")
print( confusion_matrix(Y_test, predictions))


print("\n")
print("CODE PROPOSED BY : JITENDRA PARMAR")
print("Email Id : jitendra.parmar_cs18@gla.ac.in ")

