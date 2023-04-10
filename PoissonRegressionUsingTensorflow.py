import tensorflow as tf
import numpy as np
x1=np.random.normal(1,0,(10000,))
x2=np.random.beta(2,5,(10000,))
x3=np.random.uniform(0,1,(10000,))
x=tf.reshape(tf.constant([x1,x2,x3],tf.float32),shape=(10000,3))
y=tf.reshape(tf.constant(6.3424627*x1**5+1.94127*x2**3+7.35247*x3**2+x2*0.97857+x1*4.535795+1.1212*x3+5,tf.float32),(-1,1))
class Poisson:
    def __init__(self,datax,datay,*args, **kwargs):
        self.datax=self.standartize(datax)
        self.datay=self.standartize(datay)
        self.initialize()
    def initialize(self):
        self.w=tf.Variable(tf.random.normal(dtype=tf.float32,shape=(self.datax.shape[-1],self.datay.shape[-1])))
        self.b=tf.zeros(shape=(self.datay.shape[0],),dtype=tf.float32)
    def train(self,epochs,batchSize,lr):
        for epoch in range(1,epochs+1):
            epochLoss=[]
            for batch in range(int(self.datax.shape[0]//batchSize)):
                xBatch=self.datax[batch*batchSize:min(self.datax.shape[0],batchSize*(batch+1)):]
                yBatch=self.datay[batch*batchSize:min(self.datay.shape[0],batchSize*(batch+1)):]
                vals=[self.w,self.b]
                with tf.GradientTape() as tape:
                    tape.watch(vals)
                    pred=self.predict(xBatch)
                    loss=self.meanSquaredError(pred,yBatch)
                    epochLoss.append(loss)
                gradW,gradB=tape.gradient(loss,vals)
                self.w.assign_sub(lr*gradW)
                self.b-=(lr*gradB)
            epochLoss=tf.reduce_mean(epochLoss)
            print(epochLoss)
    def predict(self,val):
        return tf.exp(tf.matmul(val,self.w)+self.b)
    def meanSquaredError(self,yT,yP):
        return tf.reduce_mean(tf.square(yT-yP))
    def standartize(self,arr):
        return (arr-tf.reduce_mean(arr))/tf.math.reduce_std(arr)
poi=Poisson(x,y)
poi.train(10000,100,1e-4)
