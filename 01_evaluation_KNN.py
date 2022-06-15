import numpy as np
from sklearn.dummy import DummyClassifier
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib  
from sklearn import metrics 
import matplotlib.pyplot as plt

wine = datasets.load_wine()
#print(wine)
wine_predictors, wine_target = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(wine_predictors, wine_target, train_size=0.7, test_size=0.3, random_state=0
)

#print(wine_target)

model = DummyClassifier()
model.fit(X_train, y_train)
dummy_score = model.score(X_test, y_test)
print('Dummy score:', dummy_score)

joblib.dump(dummy_score, 'dummy_score.pkl')
dummy_score_load = joblib.load('dummy_score.pkl')


#KNNClassifier
# k=10
# acc_test = np.zeros(k)
# acc_train = np.zeros(k)
#train_accurancy=[]
#test_accurancy=[]

def results(X_train, y_train, X_test, y_test, k=10):

    acc_test = np.zeros(k)
    acc_train = np.zeros(k)

    for k in np.arange(1,k+1, 1): 

        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train) # k changes after each iteration
        #train_accurancy.append(classifier.score(X_train, y_train))
        #test_accurancy.append(classifier.score(X_test, y_test))

        y_pred = classifier.predict(X_test)
        acc_metrics_test = metrics.accuracy_score(y_test, y_pred)
        acc_test[k-1]=acc_metrics_test

        x_pred = classifier.predict(X_train)
        acc_metrics_train = metrics.accuracy_score(y_train, x_pred)
        acc_train[k-1]=acc_metrics_train


    max_acc=np.amax(acc_test)
    acc_test_list=list(acc_test) 
    k=acc_test_list.index(max_acc)
    acc_train_list=list(acc_train)
    print("The best accuracy was with", max_acc, "with k=", k+1)
    print("Train list:", acc_test_list)
    print("Test list:", acc_train_list)

    #plot

    x = list(range(1, 11))
    plt.plot(x, acc_test, label = "Test")
    plt.plot(x, acc_train, label = "Train")
    plt.title("KNN Classifier")
    plt.legend()
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accurancy")
    plt.show()

results(X_train, y_train, X_test, y_test)

#standaryzacja 
standardizer = StandardScaler()
X_train = standardizer.fit_transform(X_train)
X_test = standardizer.fit_transform(X_test)

#results po standaryzacji
results(X_train, y_train, X_test, y_test)






