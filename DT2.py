import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics




def DTPurringAlphaBeta(param : int) :
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']
    # load info
    train = pd.read_csv("train.csv")
    x = train[features]
    y = train.Outcome
    # create the tree
    tree = DecisionTreeClassifier(criterion="entropy", min_samples_split=param,class_weight={0:1-0.8,1: 0.8})

    tree = tree.fit(x, y)
    # test the tree:
    tst = pd.read_csv("test.csv")
    x_test = tst[features]
    y_test = tst.Outcome
    y_res = tree.predict(x_test)
    mat = metrics.confusion_matrix(y_test, y_res)
    print(mat)


if __name__ == '__main__':
    DTPurringAlphaBeta(9)