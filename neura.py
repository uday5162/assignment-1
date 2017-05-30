import numpy as np
import scipy.io
from numpy import genfromtxt
X = genfromtxt('train_data.csv', delimiter=',')

temp=genfromtxt('train_labels.csv', delimiter=',')
Y=np.zeros((15000,26))

for i in range(15000):
        Y[i][temp[i]-1]=1

def sigmoid(a):
    b = (1.0)
    c = (1.0)+np.exp(-a)
    return b/c

m = len(X);
n = len(X[0]);
r1 = -0.5;
r2 = 0.5;
theta1 = (r2-r1)*np.random.rand(250,n) + r1;
theta2 = (r2-r1)*np.random.rand(26,250) + r1;
print 190
al =0;
count =0;
iter = 0;
while(iter<100):
    df = np.random.permutation(m);
    X = X[df];
    Y = Y[df];
    for i in range(m) :
        count = count +1;
        al = 1.0/np.sqrt(count);
        x = [X[i]];
        x=np.array(x);
        y = np.transpose([Y[i]]);
        a2 = sigmoid(np.dot(theta1,np.transpose(x)))
        a3 = sigmoid(np.dot(theta2,(a2)))
        del3 = (y - a3)*a3*(1-a3)
        del2 = np.dot(np.transpose(theta2),del3)*a2*(1-a2);
        del_theta1 = np.dot(del2,x);
        del_theta2 = np.dot(del3,np.transpose(a2))
        theta1 = theta1 + al*del_theta1;
        theta2 = theta2 + al*del_theta2;
    iter = iter + 1
    print "iter" ,iter

right = 0;
testx = genfromtxt('test_data.csv', delimiter=',')

testy = np.zeros((5000,26))


m_test = len(testx);
n_test = len(testx[0]);

result=[]

for i in range(m_test):
    xx = testx[i];
    yy = testy[i];
    # print "xx",xx.shape,yy.shape
    a22 = sigmoid(np.dot(theta1,np.transpose(xx)))
    a33 = sigmoid(np.dot(theta2,a22))
    # print "shape",a33.shape
    index= np.argmax(a33);
    print index, "index"
    print yy[index], "ind"
    result.append(index+1)
    if(yy[index] == 0):
        right = right + 1;
accuracy = (float(right)/(m_test))*100
print "acc",accuracy

myString = "\n".join([str(i) for i in result])
myString=myString+'\n'
fo=open("result250.csv",'w')
fo.write(myString)
