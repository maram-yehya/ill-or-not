from KNN1 import KNN
from sklearn import metrics

def sfs(k):
    features_init = list(range(8))

    final = []
    tmp  = list(range(8))

    acc=-1
    add_f=-1
    while True:
        for f in features_init :
            tmp.remove(f)
            res = KNN(k, features_index=tmp)
            y_test = res[0]
            test_res = res[1]
            new_acc = metrics.accuracy_score(y_test, test_res)
            if new_acc > acc :
                acc=new_acc
                add_f=f
            tmp.append(f)
        if add_f ==-1 :
            break
        final.append(add_f)
        features_init.remove(add_f)
        tmp.remove(add_f)
        add_f=-1
    print(final)
    print("acc: " ,acc)




if __name__ == '__main__':
    sfs(9)



