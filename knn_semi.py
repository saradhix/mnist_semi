import numpy as np
from numpy import arange
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt
test_error=[]
train_error=[]
mnist = fetch_mldata("MNIST Original")
mnist.data, mnist.target = shuffle(mnist.data, mnist.target)
n_real_train = 6000

real_train_idx=arange(0,n_real_train)
train_idx = arange(n_real_train,60000)
test_idx = arange(60000,70000)

X_real_train, y_real_train = mnist.data[real_train_idx], mnist.target[real_train_idx]
X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
print "Shape of X_real_train", X_real_train.shape
print "Shape of X_train", X_train.shape
print "Reducing dimensions using PCA with 196 components"
start_time = time.time()
pca = PCA(n_components = 196)
pca.fit(X_train)
print "Transforming training"
X_real_train_pca = pca.transform(X_real_train)
X_train_pca = pca.transform(X_train)
print "Transforming testing"
X_test_pca = pca.transform(X_test)
end_time = time.time()
print "Running time for PCA:", end_time - start_time
print "X_real_train_pca=",X_real_train_pca.shape, "X_train_pca=",X_train_pca.shape,"X_test_pca", X_test_pca.shape


for iteration in range(0,10):
  n=3
  print "Applying KNN algorithm with neighbours:", n
  print "Shape of X_real_train", X_real_train_pca.shape
  start_time = time.time()
  clf = KNeighborsClassifier(n_neighbors=n)
  clf.fit(X_real_train_pca, y_real_train)
  print "Making predictions on rest of the training data", X_train_pca.shape
  y_pred = clf.predict(X_train_pca)
  end_time = time.time()
  print "Running time for KNN:", end_time - start_time
  #Add all examples from the test which failed to train
  X_real_train_pca = X_real_train_pca.tolist()
  y_real_train = y_real_train.tolist()
  i=0
  num_predicted_wrong=0
  for (predicted, actual) in zip(y_pred, y_train):
    if predicted != actual:
      #Add this to train
      X_real_train_pca.append(X_train_pca[i])
      y_real_train.append(actual)
      num_predicted_wrong +=1
    i +=1
  percent_wrong = float(num_predicted_wrong)/i*100.0
  train_error.append(percent_wrong)
  print "Iteration %d predicted_wrong %d percent %f" % (iteration, num_predicted_wrong, percent_wrong)
  #Make numpy arrays from X_train and y_train
  X_real_train_pca = np.array(X_real_train_pca)
  y_real_train = np.array(y_real_train)
  print "Shape of training is %s" % (str(X_real_train_pca.shape))
  print "Testing on unseen test data of shape", X_test_pca.shape
  start_time = time.time()
  print X_real_train_pca.shape, y_real_train.shape
  nclf = KNeighborsClassifier(n_neighbors=n)
  nclf.fit(X_real_train_pca, y_real_train)
  print "Making predictions on unseen data"
  y_pred = nclf.predict(X_test_pca)
  end_time = time.time()
  test_mistakes = 0
  for (predicted, actual) in zip(y_pred, y_test):
    if predicted != actual:
      #Add this to train
      test_mistakes +=1
  percent_wrong = test_mistakes/100.0
  test_error.append(percent_wrong)
  print "Mistakes in test=%d percent=%f" % (test_mistakes,percent_wrong)
  print "-"*60

print train_error
print test_error
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
iterations = range(0,10)
plt.plot(iterations, train_error,iterations,test_error)
plt.show()
