import numpy 
import matplotlib.pyplot
import qpsolvers 

X= numpy.loadtxt('C:/Users/masoodhur/Desktop/Final exam6DA3/data.txt', delimiter=',')
  
y1=numpy.ones(500)
y2=-numpy.ones(500)
ylabel=numpy.concatenate((y1,y2),axis=0)

P=numpy.eye(2)
P=numpy.append(P,numpy.zeros((2,1)),axis=1)
P=numpy.append(P,numpy.zeros((1,3)),axis=0)+.00001*numpy.eye(3)

q=numpy.zeros(3)

h=-numpy.ones(1000)
G=-numpy.dot(numpy.diag(ylabel),numpy.append(X, numpy.ones((1000,1)),axis=1))


w=qpsolvers.solve_qp(P,q,G,h)

print("QP solution:",w)

slope=-w[0]/w[1]
intercept=-w[2]/w[1]


beginX=0
endX=10
beignY=intercept
endY=endX*slope+intercept
x=[beginX,endX]
y=[beignY,endY]

matplotlib.pyplot.scatter(X[0:500,0],X[0:500,1],c='r',marker="*")
matplotlib.pyplot.scatter(X[500:1000,0],X[500:1000,1],c='b',marker="*")
matplotlib.pyplot.plot(x,y,'g--')
matplotlib.pyplot.ion()
matplotlib.pyplot.show()

prediction=numpy.sign(numpy.dot(X,w[0:2]+w[2]))
error_prediction=numpy.sum(prediction!=ylabel)
#training the test data
X_test=numpy.loadtxt('C:/Users/masoodhur/Desktop/Final exam6DA3/test.txt', delimiter=',')

y1_test=numpy.ones(500)
y2_test=-numpy.ones(500)
y_test=numpy.concatenate((y1_test,y2_test),axis=0)

prediction_test=numpy.sign(numpy.dot(X_test,w[0:2]+w[2]))
num_error_prediction_test=numpy.sum(prediction_test!=y_test)
Error_rate=num_error_prediction_test/len(prediction_test)
Accuracy_rate=1-Error_rate
# calculation the Accuracy rate
TF=0
TN=0
FP=0
FN=0
for i in range(len(prediction_test)):
    if y_test[i]==1 and prediction_test[i]==1:
        TF=TF+1
    elif y_test[i]==-1 and prediction_test[i]==-1:
        TN=TN+1
    elif prediction_test[i]==1 and y_test[i]!=prediction_test[i]:
        FP=FP+1
    elif prediction_test[i]==-1 and y_test[i]!=prediction_test[i]:
        FN=FN+1
print("***********Confusion Matrix**************")       
print(TF, FP) 
print(FN, TN)

# ***************Draw the ROC Curve**********************
print("***************Draw the ROC Curve**********************")
f_score=numpy.dot(X_test,w[0:2]+w[2])
Zx, Zy = zip(*[(x, y) for x, y in sorted(zip(f_score, y_test))])

score=numpy.array(Zx)[::-1]
GT=numpy.array(Zy)[::-1]

fpr=[]
tpr=[]


P=sum(GT==1)
N=len(GT)-P

FP=0
TP=0
for i in range (len(score)):
    if(GT[i]==1):
        TP=TP+1
    if(GT[i]==-1):
        FP=FP+1
    fpr.append(FP/float(N))
    tpr.append(TP/float(P))
matplotlib.pyplot.plot(fpr, tpr)
matplotlib.pyplot.plot([0, 1],[0,1],'r--')
matplotlib.pyplot.title("ROC Curve")
matplotlib.pyplot.xlabel("False positive rate")
matplotlib.pyplot.ylabel("True Positive Rate")
matplotlib.pyplot.show()
    

