
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

dataset = pd.read_csv('handwriting.csv')

#Splitting into IVs and DPs
X = dataset.iloc[:, :62].values
Y = dataset.iloc[:, 62].values


# construct target matrix
target = np.zeros((np.shape(X)[0],10));

#1-on-N encoding
indices = np.where(Y==0)
target[indices,0] = 1
indices = np.where(Y==1)
target[indices,1] = 1
indices = np.where(Y==2)
target[indices,2] = 1
indices = np.where(Y==3)
target[indices,3] = 1
indices = np.where(Y==4)
target[indices,4] = 1
indices = np.where(Y==5)
target[indices,5] = 1
indices = np.where(Y==6)
target[indices,6] = 1
indices = np.where(Y==7)
target[indices,7] = 1
indices = np.where(Y==8)
target[indices,8] = 1
indices = np.where(Y==9)
target[indices,9] = 1

#Splitting into traning, testing and validation data
train = X[::2,:] 
traint = target[::2] # every second item (0, 2, 4...)
valid = X[1::4,:]
validt = target[1::4] # from 2nd (0-based), every fourth (1, 5, 9...)
test = X[3::4,:]
testt = target[3::4] # from 4th (0-based), every fourth (3, 7, 11...)


#Output file for experimenting with parameters
file = open("mlp.txt", "w")
file.write("nhidden," + " " + "% acc., "  + "runtime.\n")


#One hidden layer: 
#Number of runs:
runs = 1   
import mlp        
for i in range(10, 11, 2): # for testing different parameter values
    file.write("One hidden layer\n")
    start_time = time.time() #runtime for each i
    cumulative_accuracy = 0
    for x in range(1,runs+1):
        net = mlp.mlp(train,traint,nhidden=i,beta=5,outtype='softmax') # initialise network softmax
        net.earlystopping(train,traint,valid,validt,0.10, niterations=100) # train
        correct = net.confmat(test, testt)
        cumulative_accuracy += correct
    file.write(str(i) + ", " + str(cumulative_accuracy/runs) + ", " + str(float((time.time() - start_time))/runs) +"\n") 
    print("Average accuracy in "+str(runs)+" runs: " + str(cumulative_accuracy/runs))


#Two hidden layers:
import mlp2
#Number of runs:
runs = 1
for i in range(10, 11, 2):  # for testing different parameter values
    file.write("Two hidden layers\n")
    start_time = time.time() #runtime for each i
    cumulative_accuracy = 0
    for x in range(1,runs+1):
        net = mlp2.mlp2(train,traint,nhidden=i,beta=5,outtype='softmax') # initialise network softmax
        net.earlystopping(train,traint,valid,validt,0.10, niterations=100) # train
        correct = net.confmat(test, testt)
        cumulative_accuracy += correct
    file.write(str(i) + ", " + str(cumulative_accuracy/runs) + ", " + str(float((time.time() - start_time))/runs) +"\n") 
    print("Average accuracy in "+str(runs)+" runs: " + str(cumulative_accuracy/runs))


file.close()