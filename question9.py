import random
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def DT1(param) :
    features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
                'DiabetesPedigreeFunction','Age']
    #load info
    train=pd.read_csv("train.csv")
    x=train[features]
    y=train.Outcome
    #create the tree
    ID3Tree = DecisionTreeClassifier(criterion="entropy",random_state=0)
    ID3Tree = ID3Tree.fit(x,y)
    #test the tree:
    tst=pd.read_csv("test.csv")
    x_test=tst[features]

    y_test=tst.Outcome
    y_res = ID3Tree.predict(x_test)
    index=0
    while index < len(y_res):
        if y_res[index] == 0:
            flip=random.randint(0,param)
            if flip == 0:
                y_res[index] = 1
        index+=1


    #print the final  results
    mat=metrics.confusion_matrix(y_test,y_res)
    print(mat)

if __name__ == '__main__':
    DT1(4)
    DT1(9)
    DT1(19)
