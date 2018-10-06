
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import tree
from IPython.display import Image, display
from sklearn.metrics import classification_report


# In[3]:


url='https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(url)
titanic.columns


# In[4]:


titanic.head(10)


# In[5]:


cols_to_del = { 'PassengerId', 'Survived', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'}

X= titanic.drop(cols_to_del, axis=1)
Y = titanic.Survived
X.head()


# In[6]:


get_ipython().magic('matplotlib inline')
print(pd.crosstab(titanic.Sex, titanic.Survived))
pd.crosstab(titanic.Sex, titanic.Survived.astype(bool)).plot(kind= 'bar')
plt.show()


# In[9]:


print(X.Sex.unique())  #Before encoding  
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform) #check if there is more than one categorical column then how it works
X.head()


# In[10]:


dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5,random_state=2)
dtree.fit(X,Y)


# In[11]:


y_pred = dtree.predict(X)

# how did model perform?
print("Total Records in Data:",len(X))
count_misclassified = (Y != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(Y, y_pred)
print('All Data Accuracy: {:.2f}'.format(accuracy))


# In[12]:


X_train,  X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state =3)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[13]:


dtree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5,random_state=2)
dtree_model.fit(X_train,Y_train)


# In[14]:


y_train_pred = dtree_model.predict(X_train)

# how did model perform?
print("Total Records in Training Data:",len(X_train))
count_misclassified = (Y_train != y_train_pred).sum()
print('Training Data Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(Y_train, y_train_pred)
print('Training Data Accuracy: {:.2f}'.format(accuracy))


# In[15]:


y_test_pred = dtree_model.predict(X_test)

# how did model perform?
print("Total Records in Testing Data:",len(X_test))
count_misclassified = (Y_test != y_test_pred).sum()
print('Testing Data Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(Y_test, y_test_pred)
print('Testing Data Accuracy: {:.2f}'.format(accuracy))


# In[16]:


from sklearn.cross_validation import cross_val_score


# In[17]:


# Cross Validation with Entire Set of Data

#Cross Val Score Parameters (# Model to test # Target variable # Scoring metric # Cross validation folds)
scores = cross_val_score(estimator= dtree, X= X, y = Y, scoring = "accuracy", cv=10) 
print("Accuracy per fold: ")
print(scores)
print("Average accuracy: ", scores.mean())


# In[18]:


# Cross Validation with Training Set of Data

scores = cross_val_score(estimator= dtree_model, X= X_train, y = Y_train, scoring = "accuracy", cv=10) 
print("Accuracy per fold: ")
print(scores)
print("Average accuracy: ", scores.mean())


# In[19]:


# Cross Validation with Testing Set of Data

scores = cross_val_score(estimator= dtree_model, X= X_test, y = Y_test, scoring = "accuracy", cv=10) 
print("Accuracy per fold: ")
print(scores)
print("Average accuracy: ", scores.mean())

