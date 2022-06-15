#cross_valid
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.datasets import load_wine
from sklearn import linear_model, svm

wine = load_wine()
#wine.target[[0,10,1]]
#wine = wine.target_names


#KFOLD#
splits = 3
kf=KFold(n_splits=splits)
X = wine.data
y = wine.target
svc = svm.SVC(C=1, kernel='linear')


#print(len(X)//splits)
score_kf = list()
for fold_nb, (train_index, test_index) in enumerate(kf.split(X)):
    print('Folf nb:', fold_nb)
    print('train size:', len(train_index), 'test size:', len(test_index))
    print('Amount:', np.bincount(y[train_index]), 'train:', y[train_index].size, ' test:', y[test_index].size)
    scores = svc.fit(X[train_index], y[train_index])
    y_pred = svc.predict(X[test_index])
    scores = mean_absolute_error(y[test_index], y_pred)
    score_kf.append(scores)
#np.mean(score_kf)
print('Scores_kf:', score_kf)
#print('Liczebnosc:', np.bincount(X))

#StratifiedKFold#
skf = StratifiedKFold(n_splits=splits)
score_skf = []
for sfold_nb, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('SFolf nb:', sfold_nb)
    print('train size:', len(train_index), 'test size:', len(test_index))
    #print(train_index)
    print('Amount:', np.bincount(y[train_index]), 'train:', y[train_index].size, ' test:', y[test_index].size)
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    scores = svc.fit(X[train_index], y[train_index])
    y_pred = svc.predict(X[test_index])
    scores = mean_absolute_error(y[test_index], y_pred)
    score_skf.append(scores)
#np.mean(score_kf)
print('Scores_skf:', score_skf)


#CROS_VAL_SCORE
lasso = linear_model.Lasso()

for i in [3, 5, 10]:
    print('Score for dataset cv=',i,':', cross_val_score(lasso, X, y, cv = i))



