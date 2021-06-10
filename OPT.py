import math
import operator
import pandas as pd
from sklearn import metrics
from KNN1 import KNN


def opt(k):
    features_index = []
    res = KNN(k)
    y_test = res[0]
    test_res = res[1]
    acc = metrics.accuracy_score(y_test, test_res)


    tmp_acc=acc
    tmp = []
    features_init = list(range(8))
    index=0
    i=0
    while True:
        tmp.clear()
        acc=tmp_acc
        for index in features_init :
            features_index.append(index)

            res = KNN(k,features_index=features_index)
            y_test = res[0]
            test_res = res[1]
            new_acc = metrics.accuracy_score(y_test, test_res)
            tmp.append((index,new_acc))
            features_index.remove(index)
            if new_acc > acc :
                tmp_acc=new_acc
        if tmp_acc==acc :
            break

        tmp.sort(key=operator.itemgetter(1))
        features_index.append(tmp[len(tmp)-1][0])
        features_init.remove(tmp[len(tmp)-1][0])

    features_opt=list(range(8))
    features_opt=[item for item in features_opt if item not in features_index]

    res = KNN(k, features_index=features_index)
    y_test = res[0]
    test_res = res[1]
    acc = metrics.accuracy_score(y_test, test_res)

    #print("acccc",acc)
    print(features_opt)




if __name__ == '__main__':
    opt(9)