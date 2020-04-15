import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


features = pd.read_csv('riskcsv6.csv')
# print(features.head(5)

# print('The shape of our features is:', features.shape)

# print(features.describe())

labels = np.array(features['ex_pro'])
features= features.drop('ex_pro', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
# print(features)
e=features.tolist()
# print(e)



# Using Skicit-learn to split data into training and testing sets
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.1, random_state = 42)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, bootstrap= True)
rf.fit(train_features, train_labels)

y1 = rf.predict(e)
print(rf.predict(e))
