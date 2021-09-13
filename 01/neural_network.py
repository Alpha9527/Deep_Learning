'''
    搭建单层神经网络模型
code:utf-8
author:zly
date:2021-09-13
'''
import numpy as np
from numpy import exp
from numpy import log
# 激活函数
def sigmoid(x):
    y = 1/(1+exp(-x))
    return y
def relu(x):
    y = np.where(x>0,x,0)
    return y
# 损失函数
def softmax():
    pass
# x = [1,2]
# w = [[1,2],[3,4]]
# b=[0.1,-0.1]
# x =np.array(x)
# w =np.array(w)
# b =np.array(b)
# y = w*x+b
# print(x,w,b,y)
# 初始化模型参数
def model_parameter_intialization(c,dim):
    w = np.zeros((c,dim))
    b = np.zeros((c,1))
    return w,b
dim =10
c=2
w,b = model_parameter_intialization(c,dim)
print(w.shape)
x = np.zeros((dim,1))
print(x.shape)
y = np.dot(w,x)+b
print(y.shape)
print(y)
# 前向传播函数
def propagate(w,b,x,y):
    m = x.shape[0]
    A = sigmoid(np.dot(w,x)+b)
    y_p = np.dot(w,x)+b
    A = sigmoid(y_p)
    cost = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    dw = np.dot(A-y,x.T)/m
    a = sigmoid(y_p)
    A = np.sum(a)
    cost = -log()
    pass
# 反向传播函数
def backward_propagation():
    pass