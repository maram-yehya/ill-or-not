import math
import operator
import pandas as pd
from sklearn import metrics

def distance(object1, object2, features_num, features_index):
    dist = 0
    for index in range(features_num):
        if index in features_index:
            continue
        dist += pow((float(object1[index]) - float(object2[index])), 2)
    return math.sqrt(dist)

def getKNeighbors(train_objects,test, k, features_index):
    diff = []
    features_num = len(test) - 1
    for train in train_objects:
        dist = distance(train, test, features_num,features_index)
        diff.append((train, dist))
    diff.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(diff[x][0][8])
    return neighbors


def normalize():
    train = pd.read_csv("train.csv")
    features = [train.Pregnancies, train.Glucose, train.BloodPressure, train.SkinThickness, train.Insulin, train.BMI,
                train.DiabetesPedigreeFunction, train.Age]

    minmax = []


    for f in features:
        minimum = min(f)
        maximum = max(f)
        minmax.append((minimum,maximum))


    #print(train)
    lines = [line.rstrip('\n') for line in open("train.csv")]
    f = open('newfile', 'w')
    #lines = train.readlines()
    index = 0
    for line in lines:

        if index == 0:
            f.write(line+"\n")
            index += 1
            continue

        obj = line.split(",")

        for i in range(len(obj)-1):

            sub = (float(minmax[i][1]) - float(minmax[i][0]))

            obj[i] = (float(obj[i]) - float(minmax[i][0])) /sub

        obj[len(obj)-1] = (int(obj[len(obj)-1]))
        f.write(str(obj).strip('[]') + '\n')


    return minmax


def KNN(k, positive_weight = 1, features_index = []):
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']

    minmax = normalize()
    train = pd.read_csv("newfile")
    train_objects = []
    lines = [line.rstrip('\n') for line in open("newfile")]
    index=0
    for line in lines:
        if index==0:
            index+=1
            continue
        obj = line.split(",")
        for i in range(len(obj)):
            obj[i]=float(obj[i])
        train_objects.append(obj)

    test = pd.read_csv("test.csv")
    y_test = test.Outcome
    lines = [line.rstrip('\n') for line in open("test.csv")]
    test_res=[]
    index=0
    for line in lines:
        #so we dont treat the first line in the file as a test object
        if index==0:
            index+=1
            continue
        obj = line.split(",")
        for i in range(len(obj)-1):
            sub = (float(minmax[i][1]) - float(minmax[i][0]))
            obj[i] = (float(obj[i]) - float(minmax[i][0])) /sub

        obj[len(obj)-1] = (int(obj[len(obj)-1]))
        neighbors = getKNeighbors(train_objects, obj, k, features_index)

        neg = sum(1 for i in neighbors if i == 0)
        pos = sum(positive_weight for i in neighbors if i == 1)
        res=0
        if neg<pos :
            res=1

        test_res.append(res)

    # print the final  results


    return y_test, test_res

if __name__ == '__main__':
    res=KNN(9)
    y_test=res[0]
    test_res=res[1]
    mat = metrics.confusion_matrix(y_test, test_res)
    print(mat)
