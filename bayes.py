import pandas as pd
import random

data = pd.read_csv('riskcsv.csv', index_col=0)
sample = data.sample(frac=1)

sample.reset_index(drop=True)
# 重新创建索引
sample.reset_index(drop=True)
# 将采样数据存到'application_train_sample.csv'文件中
sample.to_csv('risk_sample.csv')

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

#测试
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
splitRatio = 0.94
train, test = splitDataset(dataset, splitRatio)
print('Split {0} rows into train with {1} and test with {2}'.format(len(dataset), train, test))


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i] #假设最后一个值为类别值
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

#测试
dataset = [[1,20,1], [2,21,0], [3,22,1]]
separated = separateByClass(dataset)
print('Separated instances: {0}'.format(separated))

import math


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


# 测试
numbers = [1, 2, 3, 4, 5]
print('Summary of {0}: mean={1}, stdev={2}'.format(numbers, mean(numbers), stdev(numbers)))

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

dataset = [[1,20,0], [2,21,1], [3,22,0]]
summary = summarize(dataset)
print('Attribute summaries: {0}'.format(summary))

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
summary = summarizeByClass(dataset)
print('Summary by class value: {0}'.format(summary))