# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:58:06 2020

@author: masoodhur
"""

import numpy 
import qpsolvers
import matplotlib.pyplot

mean1=numpy.random.uniform(-10,-8,10)
cov1= numpy.diag(numpy.random.uniform(0,3,10))

mean2=numpy.random.uniform(10,11,10)
cov2=numpy.diag(numpy.random.uniform(0,3,10))

x1=numpy.random.multivariate_normal(mean1,cov1,1000)
x2 = numpy.random.multivariate_normal(mean2, cov2, 1000)
X=numpy.concatenate((x1,x2),axis=0)

y = numpy.concatenate((numpy.ones(1000), -numpy.ones(1000)), axis = 0)

X=numpy.concatenate((numpy.ones((X.shape[0],1)),X),axis=1)

P=numpy.eye(10)
P=numpy.append(P,numpy.zeros((10,1)),axis=1)
P=numpy.append(P,numpy.zeros((1,11)),axis=0)+0.00001*numpy.eye(11)
q=numpy.zeros(11)


G = numpy.dot(-numpy.diag(y),X)
h = -numpy.ones(2000)

w = qpsolvers.solve_qp(P, q, G, h)
print("QP solution:", w)

xtest1 = numpy.random.multivariate_normal(mean1, cov1*30, 100)
xtest2 = numpy.random.multivariate_normal(mean2, cov2*30, 100)
Xtest = numpy.concatenate((xtest1,xtest2),axis = 0)

Xtest = numpy.concatenate((numpy.ones((Xtest.shape[0],1)),Xtest), axis = 1)

ytest = numpy.concatenate((numpy.ones(100), -numpy.ones(100)), axis = 0)

pred = numpy.sign(numpy.dot(Xtest,w))
numErrors = numpy.sum(ytest != pred)




