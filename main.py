import numpy as np
import matplotlib.pyplot as plt
x=(np.random.normal(0,1,1000)+np.random.normal(0,1,1000))/2
y=(x**5)+5.829875*(x**4)+2.387*(x**3)+(x**2)+x+4.83268
def pos(x,w):
    return np.e**(x*w)
def pos_dev(x,w):
    return x*(np.e**(x*w))
batch_size=100
epochs=300
w_=1
def loss(y_t,y_p): 
    loss=0
    for i in range(len(y_p)):
        loss+=((y_t[i]-y_p[i])**2) 
    return loss/len(y_p)
def loss_dev(x,y,y_p):
    return x*(y_p-y)
loss_his=np.array([])
for i in range(epochs):
    loss_of_epoch=0
    for j in range(int(len(x)/batch_size)):
        y_result=pos(x[j:j+batch_size:],w_)
        y_dev=loss_dev(x[j:j+batch_size:],y[j:j+batch_size:],y_result)
        loss_of_epoch+=loss(y[j:j+batch_size],y_result)
        w_=w_-0.0001*y_dev
    loss_his=np.append(loss_his,[loss_of_epoch/(int(len(x)/batch_size))])
    print(loss_of_epoch/(int(len(x)/batch_size)))

plt.plot(range(epochs),loss_his,color="red")
plt.scatter([np.where(loss_his==min(loss_his))],[min(loss_his)],color="green")
plt.show()
