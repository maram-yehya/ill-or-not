from KNN1 import KNN
from sklearn import metrics

if __name__ == '__main__':
    res = KNN(27, 4)
    y_test = res[0]
    test_res = res[1]
    mat = metrics.confusion_matrix(y_test, test_res)
    print(mat)

