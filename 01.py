import numpy as np
import pandas as pd

data = pd.read_csv('riskcsv.csv', index_col=0)
# sample = data.sample(frac=1)
# sample.reset_index(drop=False)
# 重新创建索引
# sample.reset_index(drop=True)
# 将采样数据存到'application_train_sample.csv'文件中
# sample.to_csv('risk_sample.csv')

data = pd.read_csv('riskcsv.csv', index_col=0)
my_matrix = np.loadtxt(open("riskcsv.csv","rb"),delimiter=",",skiprows=1)
# print(my_matrix)
# print(str(my_matrix))
#
x1 = my_matrix[1:15, 0:21]
print(x1)
y = my_matrix[1:15,-1]
print(y)

x2 = my_matrix[0:1, 0:21]
print(x2)


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x1, y)
print(clf.predict(my_matrix[0:1, 0:21]))
