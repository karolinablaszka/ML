#cross_valid
import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedShuffleSplit

data = load_wine()
data.target[[0,10,1]]
data = data.target_names

splits = 3
kf=KFold(n_splits=splits)
x = np.arange(0,10, 1)

print(len(x)//splits)

for fold_nb, (train_index, test_index) in enumerate(kf.split(x)):
    print('Folf nb:', fold_nb)
    print('train size:', len(train_index), 'test size:', len(test_index))

print(data)