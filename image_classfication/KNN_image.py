import cv2
import numpy as np


#CIFAR10 데이터셋 이용해서 K-nearest 이용 만들기
a=np.array((2,3))

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self,X, Y):
        self.Xtr= X
        self.Ytr= Y


    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.Ytr.dtype)
        for i in range(num_test):#행의 수만큼 for문 돌린다.
            distances = np.sum(np.abs(self.Xtr-X[i,:]), axis=1)
            min_index= np.argmin(distances)
            Ypred[i] = self.Ytr[min_index]

        return Ypred
