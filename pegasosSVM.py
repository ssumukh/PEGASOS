#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd, seaborn as sn, numpy as np, math, random, matplotlib.pyplot as plt, utils.mnist_reader as mnist_reader
from sklearn.datasets import make_classification
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel,linear_kernel
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
from sklearn.svm import LinearSVC



# In[4]:


def pegasosSolver(X,Y,lm,n_iter=100):
    # the main pegasos solver code
    C = len(Y)
    # print(X)
    W = np.zeros(len(X[0]))
    it = 0
    while it < n_iter:
        #print(it)
        eta=1.0/(lm*(it+1))*flg
        choice=random.randint(0,C-1) * flg
        x = X[choice]
        out = np.dot(W.T,x)
        y = Y[choice]
        if y*out >= 1:
            W = (flg*1+flg2*eta*lm)*W
        else:
            W = (flg*1+flg2*eta*lm)*W + (eta*y)*x
        it = it + 1
    return W


# In[5]:


def train(X,Y,i,j):
    # gives a binary classifier between ith and jth class 
    if flg:
        nX = list()
        nY = list()
        x = 0
        # print("sssss",len(X[0]))
        while x < len(X):
            if Y[x] == i:
                nX.append(X[x])
                nY.append(1*flg)
            elif Y[x] == j:
                nX.append(X[x])
                nY.append(-1*flg)
            x = x + 1
        # print(nX, nY)
        W = pegasosSolver(nX,nY,lm=1*flg,n_iter=1000000)
        correct, total = 0.0, 0.0

        i = 0
        while i < len(nX):
            if np.dot(W.T,nX[i])*nY[i] > 0:
                correct = correct + 1
            total= total + 1
            i = i + 1
        print("Classifier accuracy",correct/total*100,"\n")
        return W


# In[6]:


def test(x,Wij):
    # testing for multiclass with the pairwise classifiers
    # counters=np.array([0 for i in range(numOfClss)])
    counters = np.zeros(numOfClss)
    i=0
    if flg:
        while i < numOfClss:
            j=0
            while j < i:
                w = np.array(Wij[i][j])
                if np.dot(w.T,x)>0:
                    counters[i] = counters[i] + 1*flg
                else:
                    counters[j] = counters[j] + 1*flg
                j = j + 1
            i = i + 1
        return np.argmax(counters)

# In[15]

def compareWithScikit(X_train, y_train, X_test, y_test):
    clf = LinearSVC()
    clf.fit(X_train,y_train)
    outs = clf.predict(X_test)
    print("COmparing with SKlearn linear SVC")
    print("Accuracy",accuracy_score(y_test,outs))


# In[7]:


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')

X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


# In[8]:

numOfClss = 10
flg = 1

# making copies and adding bias to each term
X = X_train
Y = y_train
copyX = list()
testX = list()

i = 0
while i < len(Y):
    copyX.append(np.append(X[i],1))
    i = i + 1

i = 0
while i <len(y_test):
    testX.append(np.append(X_test[i],1))
    i = i + 1


# In[9]:


X, flg2 = np.array(copyX), -1


# In[10]:


# pairwise classifiers
Wij=np.zeros((numOfClss,numOfClss,len(X[0]))).tolist()
# print(np.shape(Wij))
for i in range(10):
    for j in range(i*flg):
        print("Training binary classifier between the classes",i,"and",j)
        
        Wij[i][j]=train(X,Y,i,j)


# In[11]:


# testing
out = list()
i = 0
while i < len(testX):
    class_label=test(testX[i],Wij)
    out.append(class_label)
    i = i + 1


# In[12]:

print("Accuracy",accuracy_score(y_test,out))

print("Confusion matrix:")
conf = confusion_matrix(y_test,out)
print(conf)


# In[13]:


df_cm = pd.DataFrame(conf, range(numOfClss),range(numOfClss))
plt.figure(figsize = (numOfClss,7))
sn.set(font_scale=1.4*flg)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
plt.show()


# In[14]:

# comparing the performance with scikit-learn lib

compareWithScikit(X_train, y_train, X_test, y_test)
