import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def DT1() :
    features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
                'DiabetesPedigreeFunction','Age']
    #load info
    train=pd.read_csv("train.csv")
    x=train[features]
    y=train.Outcome
    #create the tree
    ID3Tree = DecisionTreeClassifier(criterion="entropy")
    ID3Tree = ID3Tree.fit(x,y)
    #test the tree:
    tst=pd.read_csv("test.csv")
    x_test=tst[features]

    y_test=tst.Outcome
    y_res = ID3Tree.predict(x_test)

    #print the final  results
    mat=metrics.confusion_matrix(y_test,y_res)
    print(mat)

def DTPurring(param : int) :
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']
    # load info
    train = pd.read_csv("train.csv")
    x = train[features]
    y = train.Outcome
    # create the tree
    ID3Tree = DecisionTreeClassifier(criterion="entropy", min_samples_split=param)
    ID3Tree = ID3Tree.fit(x, y)
    # test the tree:
    tst = pd.read_csv("test.csv")
    x_test = tst[features]
    y_test = tst.Outcome
    y_res = ID3Tree.predict(x_test)
    # print the accuracy of the solution :
    #print("Accuracy for x=", param, " : ", metrics.accuracy_score(y_test, y_res))


def visulizeTree2(param: int):
    from sklearn.tree import export_graphviz
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']
    train = pd.read_csv("train.csv")
    x = train[features]
    y = train.Outcome
    # create the tree
    ID3Tree = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=param)
    ID3Tree = ID3Tree.fit(x, y)

    export_graphviz(ID3Tree, out_file='tree_nonlimited.dot', feature_names=features, filled=True, rounded=True,
                    special_characters=True)


if __name__ == '__main__':
    DT1()


