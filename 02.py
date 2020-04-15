import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.naive_bayes import MultinomialNB




csv_file = "riskcsv2.csv"
csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
# print(csv_df)

# dataframe to matrix
# print(csv_df.values)

for num in range(1,16):
    test = csv_df[(csv_df['id'] == num)]
    # print(test)

    train = csv_df[(csv_df['id'] != num)]
    # print(train)

    cla = train['ex_pro']
    # print(cla)

    test_m = test.values
    print(test_m)

    train_m = train.values
    # print(train_m)

    cla_m = cla.values
    # print(cla_m)

    x1 = train_m[0:14, 0:26]
    # print(x1)
    y = cla_m[0:14]
    # print(y)
    #
    clf = MultinomialNB()
    clf.fit(x1, y)
    print(clf.predict(test_m[0:14, 0:26]))
    #
