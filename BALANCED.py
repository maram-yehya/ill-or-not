# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation


def balance():
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']
    postive = 0
    train = pd.read_csv("train.csv")
    y = train.Outcome

    for l in y:
        if l == 1:
            postive += 1
    neg = 0
    lines = [line.rstrip('\n') for line in open("train.csv")]
    f = open('newfile', 'w')

    for line in lines:
        length = len(line)
        if neg >= postive and line[length - 1] == "0":
            lines.remove(line)
            continue

        if line[length - 1] == '0':
            neg += 1
        f.write(line + "\n")

    f = open('newfile', 'r')

    train = pd.read_csv("newfile")
    x = train[features]
    y = train.Outcome
    ID3Tree = DecisionTreeClassifier(criterion="entropy")
    ID3Tree = ID3Tree.fit(x, y)
    # test the tree:
    tst = pd.read_csv("test.csv")
    x_test = tst[features]
    y_test = tst.Outcome
    y_res = ID3Tree.predict(x_test)
    # print the final  results
    mat = metrics.confusion_matrix(y_test, y_res)
    print(mat)


if __name__ == '__main__':
    balance()