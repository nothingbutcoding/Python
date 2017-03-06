
# coding: utf-8

# In[1]:

import os
os.path.abspath("")


# In[10]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd

train=pd.read_csv('train.csv',index_col=0)


# In[29]:


x_train=train[train.columns[0:-1]].values
y_train=train['Cover_Type'].values


# In[23]:

y_train[1]


# In[26]:


len(y_train)


# # Decision Tree

# In[30]:

from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier (max_depth=30)
scores=cross_val_score(clf,x_train, y_train, cv=5, n_jobs=5)

print np.mean(scores), "+/1", np.std(scores)


# # Random Forest

# In[33]:

import random
random.seed(10)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()

scores=cross_val_score(clf, x_train, y_train, cv=5, n_jobs=5)

print np.mean(scores), "+/1", np.std(scores)


# In[35]:

clf.fit(x_train,y_train)

feature_importances =zip(train.columns, (clf.feature_importances_*100).astype(int))

sorted(feature_importances, key=lambda x: -x[1])[0:10]


# # SVM

# In[37]:

from sklearn.svm import SVC

clf=SVC(C=1000)

scores =cross_val_score(clf, x_train, y_train, cv=5, n_jobs=5)

print np.mean(scores), "+/1", np.std(scores)


# ## Feature Scaling

# In[39]:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

clf = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=1000))])
scores =cross_val_score(clf, x_train, y_train, cv=5, n_jobs=5)
print np.mean(scores), "+/1", np.std(scores)


# # Ensemble

# In[40]:

import random
random.seed(10)


from sklearn.ensemble import BaggingClassifier


clf = BaggingClassifier (n_estimators=100)
clf = Pipeline([('scaler', StandardScaler()), ('clf',clf)])
scores =cross_val_score(clf, x_train, y_train, cv=5, n_jobs=5)
print np.mean(scores), "+/1", np.std(scores)

