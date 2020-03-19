#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

#preprocessing

def label(x):
    if x=='Male':
        return 1
    return 0

actual_genders = dataset['Gender']
encoded_genders = actual_genders.map(label)
dataset['Gender']= encoded_genders


#fearure scaling on age and estimated salary
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
dataset_scaling_features= sc.fit_transform(dataset.loc[:,"Age": "EstimatedSalary"])

dataset["Age"]= dataset_scaling_features[:,0]
dataset["EstimatedSalary"]= dataset_scaling_features[:,1]


X=dataset.iloc[:,1:4].values
Y= dataset.iloc[:,4].values

# using onehot encoder for gender feature
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0]) 
X=onehotencoder.fit_transform(X).toarray()

#deleting used variable
del (actual_genders)
del (dataset)
del (encoded_genders)
del (dataset_scaling_features)

#splitting data into test and training set
from sklearn.model_selection import train_test_split
X_train, X_test , Y_train, Y_test =train_test_split(X,Y,test_size= 0.2, random_state= 0)


#fitting classifier
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5, algorithm = 'auto' )
neigh.fit(X_train,Y_train)

#predicting the test set result
Y_pred = neigh.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)



from sklearn.metrics import f1_score
f1_score(Y_test, Y_pred, average='macro')



