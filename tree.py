import pandas as pd
from sklearn import tree

csv_file = "riskcsv2.csv"
csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
# print(csv_data)
csv_df = pd.DataFrame(csv_data)
# print(csv_df)
# print(csv_df.values)
x2 = csv_df.values
x3 = x2[:,0:-1]

y2 = x2[:, -1]



for num in range(1,16):
    test = csv_df[(csv_df['id'] == num)]
    # print(test)

    train = csv_df[(csv_df['id'] != num)]
    # print(train)

    cla = train['ex_pro']
    # print(cla)

    test_m = test.values
    # print(test_m)

    train_m = train.values
    # print(train_m)

    cla_m = cla.values
    # print(cla_m)

    x1 = train_m[0:14, 0:26]
    # print(x1)
    y = cla_m[0:14]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x1, y)



tree.plot_tree(clf.fit(x3, y2))
