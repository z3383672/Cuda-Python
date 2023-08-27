import numpy as np
import os
import pandas
from datetime import datetime
from scipy import special
os.chdir(r'C:\Users\mhosseini\Desktop\Python app')
from datetime import timedelta
import networkx as nx
df = pandas.read_csv('file:///C:/Users/mhosseini/Desktop/Python app/purchase_time_purchase_purchasecount_purchasebefore_nousd_userid_purchaseocunt_addonetogetaccuratepurchasecount.csv')
df=df.sort_values(['pramount','totalpurchasebefore'], ascending=[True, True])
x=range(0,df.shape[0])
df['indexx']=x
x=np.zeros(df.shape[0])
df['maxx']=x
df['minn']=x


#for i in range(df.shape[0]):
#    k1 = df.loc[(df.pramount == df.iloc[i].pramount2) & (df.totalpurchasebefore == (df.iloc[i].pramount+df.iloc[i].totalpurchasebefore))]
#    if(k1.shape[0]>0):
#        df.at[i,'maxx']=np.max(k1.index.values)
#        df.at[i,'minn']=np.min(k1.index.values)

df['totalpluspr']=df['totalpurchasebefore']+df['pramount']
df2=df[['totalpurchasebefore','pramount','indexx']]
df2.columns=['totalpluspr','pramount2', 'indexx']
#df2=df2.loc[(df2['total+pr']>50)]


#df3=pandas.SparseDataFrame(df2,index=['total+pr','pramount2'], columns='col')
#
#df2['col'] = 's' + df2.groupby(['total+pr','pramount2'])['indexx'].cumcount().astype(str)
#df2 = pandas.pivot_table(df2,index=['total+pr','pramount2'], columns='col',values='indexx').reset_index()
#result = pandas.merge(df, df2,how='left')
#result=result.fillna(0)
#
#result['maxx']=np.max(result[])

#from scipy.sparse import csr_matrix
#from numpy import sort
#
#
#totalpluspr = list(sort(df2.totalpluspr.unique()))
#pramount2 = list(sort(df2.pramount2.unique()))
#
#data = df2['indexx'].tolist()
#row = df2.totalpluspr.astype('category', categories=totalpluspr).cat.codes
#col = df2.pramount2.astype('category', categories=pramount2).cat.codes
#sparse_matrix = csr_matrix((data, (row, col)), shape=(len(totalpluspr), len(pramount2)))
#dfs=pandas.SparseDataFrame([ pandas.SparseSeries(sparse_matrix[i].toarray().ravel(), fill_value=0) 
#                              for i in np.arange(sparse_matrix.shape[0]) ], index=totalpluspr, columns=pramount2, default_fill_value=0)


x=df2[['totalpluspr', 'pramount2','indexx']].groupby(['totalpluspr', 'pramount2']).max()['indexx']
y=df2[['totalpluspr', 'pramount2','indexx']].groupby(['totalpluspr', 'pramount2']).min()['indexx']
x=x.reset_index()
y=y.reset_index()
x.columns=['totalpluspr','pramount2', 'indexx_max']
y.columns=['totalpluspr','pramount2', 'indexx_min']
b=pandas.merge(df, x,how='left')
b=pandas.merge(b, y,how='left')
b=b.fillna(0)
b['maxx']=b['indexx_max']
b['minn']=b['indexx_min']
b=b[['UserID','GamingServerID','pramount','time','purchasecount','totalpurchasebefore','pramount2','indexx','maxx','minn']]
df=b

data = df.values
data=np.delete(data,0,1)
data=np.delete(data,0,1)

np.savetxt('x.csv', x,delimiter=",")

data = np.loadtxt('test1.txt', dtype=int)
#Datatokeep=data

#df.to_pickle('numba_purchase_adjusted.pkl')    
#df = pandas.read_pickle('numba_purchase_adjusted.pkl')  
    
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



#k1 = df.loc[(df.pramount == 80.0) & (df.totalpurchasebefore == 0.0)]
#start=np.min(k1.index.values)
#finish=np.max(k1.index.values)
total_purchase=df.iloc[i]['totalpurchase']
pramount=df.iloc[i]['pramount']
time=df.iloc[i]['time to pass']


total_purchase=3635
pramount=80
time=0



def estimate_purchase(total_purchase,pramount,time):
    #k1=data[np.where((data[:,0]==pramount) & (data[:,4]==total_purchase))]
    k1=data[np.where((data[:,0]>= (0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount)))& (data[:,3]>=(0.9*total_purchase)) & (data[:,3]<=(1.1*total_purchase)) & (data[:,1]>=time))]
    if(k1.shape[0]>15):
        threads_per_block = 512
        blocks = 512
        seed=np.random.randint(1,100)
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed)
        out = np.zeros(threads_per_block * blocks, dtype=np.complex64)
        Monte_carlo_simulation_2[blocks, threads_per_block](data,k1,rng_states,14,-time,out)
        #np.quantile(np.imag(out),[0, 0.01, 0.02, 0.03, 0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        return (np.quantile(np.imag(out),[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
    else:
        return (np.quantile(-1,[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
    

x=result
df = pandas.read_csv('troubled_customer.csv')
result=[]
for i in range(df.shape[0]):
    result.append(estimate_purchase(df.iloc[i]['totalpurchase'],df.iloc[i]['pramount'],df.iloc[i]['time to pass']))
    print(i)

minn=[]
for i in range(df.shape[0]):
    if(type(result[i]) is np.ndarray):
        minn.append(result[i][0])
    else:
        minn.append(-1)

import csv
with open("output.csv",'w',newline='') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(result)




#######################################################

np.savetxt("purchase.csv", purchase, delimiter=",")

np.savetxt("time.csv", time, delimiter=",")


from numba.cuda.random import create_xoroshiro128p_states


@cuda.jit
def simulation(k1,rng_states,out):
    s=1508803
    f=1508806
    t=-0
    thread_id = cuda.grid(1)
    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*(k1.shape[0]))
    out[thread_id]=t+k1[randomidx,1]


threads_per_block = 512
blocks = 512 

rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
out = np.zeros(threads_per_block * blocks, dtype=np.float32)

simulation[blocks, threads_per_block](k1,rng_states, out)
print(out.reshape(blocks,threads_per_block))

###########################################################