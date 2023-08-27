import numpy as np
import os
import pandas
from datetime import datetime
from scipy import special
os.chdir(r'C:\Users\mhosseini\Desktop\Python app')
from datetime import timedelta
import networkx as nx
df = pandas.read_csv('purchase_time_purchase_purchasecount_purchasebefore.csv')
d=df.values



from numba import jit
import math
@jit(nopython=True)
def hypot(x,y):
    x=abs(x)
    y=abs(y)
    t=min(x,y)
    x=max(x,y)
    t=t/x
    return x*math.sqrt(1+t*t)

%timeit hypot(3.0,4.0)



%timeit hypot.py_func(3,4)

@jit(nopython=True)
def ex1(x,y,out):
    for i in range(0,x.shape[0]):
        out[i]=hypot(x[i],y[i])


in1=np.arange(10,dtype=np.float64)
in2=2*in1+1
out=np.empty_like(in1)

print("in1:",in1)
print("in2:",in2)

ex1(in1,in2,out)

print('out:',out)


np.testing.assert_almost_equal(out,np.hypot(in1,in2))



a=np.array([1,2,3,4])
b=np.array([10,20,30,40])

np.add(a,b)


from numba import vectorize,int64,float64,int32,float32,cuda,int16

@vectorize([int32(int32,int32)],target='cuda')
def add_ufunc(x,y):
    return x+y


%timeit add_ufunc(a,b)


%timeit np.add(a,b)

from math import exp

SQRT_2PI=np.float32((2*math.pi)**0.5)

@vectorize([float64(float32,float32,float32)],target='cuda')
def gaussian_pdf(x,mean,sigma):
    return exp(-0.5*((x-mean)/sigma)**2)/(sigma*SQRT_2PI)


x=np.random.uniform(-3,3,size=1000000000).astype(np.float32)
mean=np.float32(0.0)
sigma=np.float32(1.0)

%timeit gaussian_pdf(x,0.0,1.0)


import scipy.stats
norm_pdf=scipy.stats.norm
%timeit norm_pdf.pdf(x,loc=mean,scale=sigma)



from numba import cuda
@cuda.jit(device=True)
def polar_to_cartesian(rho,theta):
    x=rho*math.cos(theta)
    y=rho*math.sin(theta)
    return x,y

@vectorize([float32(float32,float32,float32,float32)],target='cuda')
def polar_distance(rho1,theta1,rho2,theta2):
    x1,y1=polar_to_cartesian(rho1,theta1)
    x2,y2=polar_to_cartesian(rho2,theta2)
    return ((x1-x2)**2+(y1-y2)**2)**0.5


n=100000000
rho1=np.random.uniform(0.5,1.5,size=n).astype(np.float32)
rho2=np.random.uniform(0.5,1.5,size=n).astype(np.float32)

theta1=np.random.uniform(-np.pi,np.pi,size=n).astype(np.float32)
theta2=np.random.uniform(-np.pi,np.pi,size=n).astype(np.float32)

polar_distance(rho1,theta1,rho2,theta2)



%matplotlib inline
from matplotlib import pyplot as plt

n=1000000
noise=np.random.normal(size=n)*3
pulses=np.maximum(np.sign(np.arange(n)/(n/23))-0.3,0.0)
waveform=((pulses*300)+noise).astype(np.int16)

import math

@vectorize([int16(int16,int16)],target='cuda')
def zero_suppress(waveform_value,threshold):
    if(waveform_value<threshold):
        result=0
    else:
        result=waveform_value
    return result



zero_suppress(waveform,15.0)


@vectorize([float32(float32,float32)],target='cuda')
def add_ufunc(x,y):
    return x+y


n=100000
x=np.arange(n).astype(np.float32)
y=2*x

%timeit add_ufunc(x,y)



x_device=cuda.to_device(x)
y_device=cuda.to_device(y)

print(x_device)
print(x_device.shape)
print(x_device.dtype)

%timeit add_ufunc(x_device,y_device)


output_device=cuda.device_array(shape=(n,),dtype=np.float32)

%timeit add_ufunc(x_device,y_device,out=output_device)
out_host=output_device.copy_to_host()
print(out_host)



from numba import cuda



@cuda.jit
def add_kernel(x,y,out):
    tx=cuda.threadIdx.x
    ty=cuda.blockIdx.x
    
    block_size=cuda.blockDim.x
    grid_size=cuda.gridDim.x
    
    start=tx+ty*block_size
    stride=block_size*grid_size
    
    for i in range(start,x.shape[0],stride):
        out[i]=x[i]+y[i]

import numpy as np
n=10000000
x=np.arange(n).astype(np.float32)
y=2*x
out=np.empty_like(x)

thread_per_block=128
blocks_per_grid=30
%timeit add_kernel[blocks_per_grid,thread_per_block](x,y,out)
print(out[:10])




@cuda.jit
def add_kernel(x,y,out):
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for i in range(start,x.shape[0],stride):
        out[i]=x[i]+y[i]

x_device=cuda.to_device(x)
y_device=cuda.to_device(y)
out_device=cuda.device_array_like(x)

%timeit add_kernel[blocks_per_grid,thread_per_block](x,y,out)

%timeit add_kernel[blocks_per_grid,thread_per_block](x_device,y_device,out_device)


@cuda.jit
def thread_counter_race_condition(global_counter):
    global_counter[0]+=1
    

@cuda.jit
def thread_counter_safe(global_counter):
    cuda.atomic.add(global_counter,0,1)
    
global_counter=cuda.to_device(np.array([0],dtype=np.int32))
thread_counter_race_condition[64,64](global_counter)
print('Should be %d:' % (64*64), global_counter.copy_to_host())

global_counter = cuda.to_device(np.array([0], dtype=np.int32))
thread_counter_safe[64, 64](global_counter)

print('Should be %d:' % (64*64), global_counter.copy_to_host())







def cpu_histogram(x,xmin,xmax,histogram_out):
    nbins=histogram_out.shape[0]
    bin_width=(xmax-xmin)/nbins
    
    for element in x:
        bin_number=np.int32((element-xmin)/bin_width)
        if bin_number >=0 and bin_number < histogram_out.shape[0]:
            histogram_out[bin_number]+=1

x=np.random.normal(size=10000,loc=0,scale=1).astype(np.float32)
xmin=np.float32(-4.0)
xmax=np.float32(4.0)
histogram_out=np.zeros(shape=10,dtype=np.int32)

%timeit cpu_histogram(x,xmin,xmax,histogram_out)

histogram_out

import numba

@numba.jit(nopython=True)
def compute_bin(x,n,xmin,xmax):
    if x==xmax:
        return n-1
    bin=np.int32(n*(np.float32(x)-np.float32(xmin))/(np.float32(xmax)-np.float32(xmin)))
    
    if bin<0 or bin >=n:
        return None
    else:
        return bin


@cuda.jit
def cuda_histogram(x, xmin, xmax, histogram_out):
    nbins=histogram_out.shape[0]
    bin_width=(xmax-xmin)/nbins
    start=cuda.grid(1)
    stride=cuda.gridsize(1)
    for i in range(start,x.shape[0],stride):
        bin_number=compute_bin(x[i],nbins,xmin,xmax)
        if bin_number >=0 and bin_number <histogram_out.shape[0]:
            cuda.atomic.add(histogram_out,bin_number,1)
            
    
    
%timeit cuda_histogram(x,xmin,xmax,histogram_out)


x_device=cuda.to_device(x)
xmin_device=cuda.to_device(xmin)
xmax_device=cuda.to_device(xmax)
histogram_out_device=cuda.to_device(histogram_out)

%timeit cuda_histogram(x_device,xmin_device,xmax_device,histogram_out_device)


from numba import curand

from numba import cuda
print(cuda.gpus)

@cuda.jit
def my_kernel(io_array):
#    tx=cuda.threadIdx.x
#    ty=cuda.blockIdx.x
#    bw=cuda.blockDim.x
    
#    pos=bw*ty+tx
    x,y=cuda.grid(2)
    if (x<io_array.shape[0]) & (y<io_array.shape[1]):
        io_array[x,y]*=2

import numpy
import math
data=numpy.ones((16,16))
threadperblock=(16,16)
blockpergridx=math.ceil(data.shape[0]/threadperblock[0])
blockpergridy=math.ceil(data.shape[1]/threadperblock[1])
blockpergrid=(blockpergridx,blockpergridy)

my_kernel[blockpergrid,threadperblock](data)

print(data)

###Matrix Multiplication


@cuda.jit
def matrix_multiplication(A,B,C):
    x,y=cuda.grid(2)
    if x<C.shape[0] & y< C.shape[1]:
        tmp=0
        for i in range(A.shape[1]):
            tmp+=A[x,i]*B[i,y]
        C[x,y]=tmp
                
A=numpy.full((24,12),3,numpy.float)
B=numpy.full((12,22),3,numpy.float)

A_global_mem=cuda.to_device(A)
B_global_mem=cuda.to_device(B)

C_global_mem=cuda.device_array((24,22))


threadperblock=(16,16)
blockpergridx=math.ceil(A.shape[0]/threadperblock[0])
blockpergridy=math.ceil(B.shape[1]/threadperblock[1])
blockpergrid=(blockpergridx,blockpergridy)

%timeit matrix_multiplication[blockpergrid,threadperblock](A_global_mem,B_global_mem,C_global_mem)




####fast multiplication
TPB=16
from numba import cuda, float32
import numpy
@cuda.jit
def fast_multipication(A,B,C):
    
    sA=cuda.shared.array(shape=(TPB,TPB),dtype=float32)
    sB=cuda.shared.array(shape=(TPB,TPB),dtype=float32)
    
    x,y=cuda.grid(2)
    
    tx=cuda.threadIdx.x
    ty=cuda.threadIdx.y
    
    if x>=C.shape[0] & y>=C.shape[1]:
        return
    
    tmp=0
    for i in range(int(A.shape[1]/TPB)):
        sA[tx,ty]=A[x,ty+i*TPB]
        sB[tx,ty]=B[tx+i*TPB,y]
    
        cuda.syncthreads()
        
        for j in range(TPB):
            tmp+=sA[tx,j]*sB[j,ty]
            
        cuda.syncthreads()
    
    C[x,y]=tmp

A=numpy.full((TPB*2,TPB*3),3,numpy.float)
B=numpy.full((TPB*3,TPB*1),4,numpy.float)


A_global_mem=cuda.to_device(A)
B_global_mem=cuda.to_device(B)

C_global_mem=cuda.device_array((TPB*2,TPB*1))

threadperblock=(TPB,TPB)
blockpergridx=int(math.ceil(A.shape[0]/threadperblock[1]))
blockpergridy=int(math.ceil(B.shape[1]/threadperblock[0]))
blockpergrid=(blockpergridx,blockpergridy)

%timeit fast_multipication[blockpergrid,threadperblock](A_global_mem,B_global_mem,C_global_mem)

res = C_global_mem.copy_to_host()

print(res)































