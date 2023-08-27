import numpy as np
import os
import pandas
from datetime import datetime
from scipy import special
os.chdir(r'C:\Users\mhosseini\Desktop\Python app')
from datetime import timedelta
import networkx as nx
df = pandas.read_csv('purchase_time_purchase_purchasecount_purchasebefore_nousd_userid_purchaseocunt_dayfromstart.csv')
df['r']=df['totalpurchasebefore']/(df['dayfromstart']+0.01)
df['r']=np.round(df['r'])

df=df.sort_values(['pramount','totalpurchasebefore','r'], ascending=[True, True,True])
x=range(0,df.shape[0])
df['indexx']=x
x=np.zeros(df.shape[0])
df['maxx']=x
df['minn']=x
df['totalpluspr']=df['totalpurchasebefore']+df['pramount']
df['rplus']=np.round((df['totalpurchasebefore']+df['pramount'])/(df['dayfromstart']+df['time']))

df2=df[['totalpurchasebefore','pramount','r','indexx']]
df2.columns=['totalpluspr','pramount2','rplus','indexx']
#df2=df2.loc[(df2['total+pr']>50)]

x=df2[['totalpluspr', 'pramount2','rplus','indexx']].groupby(['totalpluspr', 'pramount2','rplus']).max()['indexx']
y=df2[['totalpluspr', 'pramount2','rplus','indexx']].groupby(['totalpluspr', 'pramount2','rplus']).min()['indexx']
x=x.reset_index()
y=y.reset_index()
x.columns=['totalpluspr','pramount2','rplus','indexx_max']
y.columns=['totalpluspr','pramount2','rplus','indexx_min']
b=pandas.merge(df, x,how='left')
b=pandas.merge(b, y,how='left')
b=b.fillna(0)
b['maxx']=b['indexx_max']
b['minn']=b['indexx_min']
b=b[['UserID','GamingServerID','pramount','time','totalpurchasebefore','pramount2','r','indexx','maxx','minn']]
df=b

data = df.values
data=np.delete(data,0,1)
data=np.delete(data,0,1)

np.savetxt('x.csv',x,delimiter=",")

data = np.loadtxt('test1.txt', dtype=int)
 
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32   
from numba import cuda

@cuda.jit
def Monte_carlo_simulation(data,start,finish,rng_states,time,out):
    t=0
    thread_id = cuda.grid(1)
    total=0
    s=start
    f=finish+1
    while t<time:
        randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*(f-s)+s)
        if((t+data[randomidx,1])<=time):
            t=t+data[randomidx,1]
            total=total+data[randomidx,4]
            s=data[randomidx,7]
            f=data[randomidx,6]+1
            if (data[randomidx,4]==0):
                break
        else:
            break
    out[thread_id]=complex(t,total)
    
    
    


@cuda.jit
def Monte_carlo_simulation_2(data,k1,rng_states,time,gap,out):
    t=gap
    thread_id = cuda.grid(1)
    total=0
    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*k1.shape[0])
    #row=np.int32(k1[randomidx][5])
    if((t+k1[randomidx,1])<=time):
            t=t+k1[randomidx,1]
            total=total+k1[randomidx,4]
            s=k1[randomidx,7]
            f=k1[randomidx,6]+1
            if (k1[randomidx,4]==0):
                out[thread_id]=complex(t,total)
            else:
                while t<time:
                    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*(f-s)+s)
                    if((t+data[randomidx,1])<=time):
                        t=t+data[randomidx,1]
                        total=total+data[randomidx,4]
                        s=data[randomidx,7]
                        f=data[randomidx,6]+1
                        if (data[randomidx,4]==0):
                            break
                    else:
                        break
                out[thread_id]=complex(t,total)
    else:
        out[thread_id]=complex(t,total)  
        


@cuda.jit
def Monte_carlo_simulation_3(data,k1,rng_states,time,gap,out):
    t=gap
    thread_id = cuda.grid(1)
    total=0
    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*k1.shape[0])
    if((t+k1[randomidx,1])<=time):
            t=t+k1[randomidx,1]
            total=total+k1[randomidx,3]
            s=k1[randomidx,7]
            f=k1[randomidx,6]+1
            if (k1[randomidx,3]==0):
                out[thread_id]=complex(t,total)
            elif((k1[randomidx,7]==0) & (k1[randomidx,6]==0)):
                out[thread_id]=complex(t,-1.0)
            else:
                while t<=time:
                    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*(f-s)+s)
                    if((t+data[randomidx,1])<=time):
                         t=t+data[randomidx,1]
                         total=total+data[randomidx,3]
                         s=data[randomidx,7]
                         f=data[randomidx,6]+1
                         if((data[randomidx,7]==0) & (data[randomidx,6]==0)):
                            total=-1.0
                            break
                         elif(data[randomidx,3]==0):
                            break
                    else:
                        break
                out[thread_id]=complex(t,total)
    else:
        out[thread_id]=complex(t,total)          
#4394
#        
#total_purchase=7268
#pramount=13
#time=0
#r=35      
#k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)) & (data[:,1]>=time) & (data[:,4]>=(0.9*r)) & (data[:,4]<=(1.1*r)))]
#       
#k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)) & (data[:,1]>=time))]
#    
#
#total_purchase=df.iloc[i]['totalpurchase']
#pramount=df.iloc[i]['pramount']
#time=df.iloc[i]['time to pass']
#purchasecount=df.iloc[i]['purchasecount']

def estimate_purchase(total_purchase,pramount,time,ratio):
    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)) & (data[:,1]>=time) & (data[:,4]>=(0.9*ratio)) & (data[:,4]<=(1.1*ratio)))]
    if(k1.shape[0]<15):
        k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) & (data[:,3]<=(1.1*total_purchase)) & (data[:,1]>=time) & (data[:,2]>=(0.9*purchasecount)) & (data[:,2]<=(1.1*purchasecount)))]
    if(k1.shape[0]>15):
        threads_per_block = 1024
        blocks = 512
        seed=np.random.randint(1,100)
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed)
        out = np.zeros(threads_per_block * blocks, dtype=np.complex64)
        Monte_carlo_simulation_3[blocks, threads_per_block](data,k1,rng_states,14,-time,out)
        #np.quantile(np.imag(out),[0, 0.01, 0.02, 0.03, 0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        return (np.quantile(np.imag(out),[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
    else:
        return (np.quantile(-1,[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
    


df = pandas.read_csv('userid_no_usdpurchase_day_from_start_canadina_dropoff.csv')
df['r']=df['totalpurchasebefore']/(df['dayfromstart']+0.01)
df['r']=np.round(df['r'])

result=[]
for i in range(df.shape[0]):
    result.append(estimate_purchase(df.iloc[i]['totalpurchase'],df.iloc[i]['pramount'],df.iloc[i]['time to pass'],df.iloc[i]['r']))
    print(i)

minn=[]
for i in range(df.shape[0]):
    if(type(result[i]) is np.ndarray):
        minn.append(result[i][0])
    else:
        minn.append(-1)

import csv
with open("output.csv",'w') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(w)


######################################################3
w=[]



t=0
total=0
time=14
import random
index = np.random.choice(k1.shape[0], 1, replace=True)
print(k1[index,])
w.append(k1[index,]) 
if((t+k1[index,1])<=time):
    t=t+k1[index,1]
    total=total+k1[index,4]
    s=k1[index,7]
    f=k1[index,6]+1
    if (k1[index,4]==0):
        print('--------')
    else:
        while t<=time:
            index=np.int32(random.uniform(0, 1)*(f-s)+s)
            print(data[index,])
            w.append(data[index,]) 
            if((t+data[index,1])<=time):
                t=t+data[index,1]
                total=total+data[index,4]
                s=data[index,7]
                f=data[index,6]+1
                if (data[index,4]==0):
                    print('--------')
                    break
            else:
                break
print(total)

       


#######################################################

np.savetxt("purchase.csv", purchase, delimiter=",")

np.savetxt("time.csv", time, delimiter=",")


from numba.cuda.random import create_xoroshiro128p_states


@cuda.jit
def simulation(rng_states,out):
    s=1508803
    f=1508806
    thread_id = cuda.grid(1)
    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*(f-s)+s)
    out[thread_id]=randomidx


threads_per_block = 4
blocks = 4 
rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
out = np.zeros(threads_per_block * blocks, dtype=np.float32)

simulation[blocks, threads_per_block](rng_states, out)
print(out.reshape(blocks,threads_per_block))

###########################################################