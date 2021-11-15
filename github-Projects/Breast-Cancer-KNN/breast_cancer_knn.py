import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate



#First one is the actual data that corresponds to the breast cancer set
#The second one is gives you a little bit more information on the dataset itself(Actually we dont need to download this)

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df.drop(['class']))

X_train, X_test, y_train, y_test = cross_validate.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)