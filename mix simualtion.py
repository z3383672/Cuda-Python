#the main idea of writing thsi version is about appreciation of the simialrity ofthe purchasing ebhaviour 
#for example between last three week and next 10 days almost 80 percent of teh time(this percenatge will be estimated more deeply)
#so instead of running the simulation ergardless of teh pattern of purchase in last three week I am going to generate two different simulation. the firts 
#is the genral simualtion that regardles of last for example three week spurchase pattern just only using the totak purchase
#and prmaount predcit the purchase value for exampel in next 10 days. the other simulation use the last three weeks purchae pattern 
#to create purchase pattern simialr to last three week(these are the time that purchase will happen) and then use the previous simualtio
#data such as total purchase and pramount to predict teh next purchase value. Then I am going to create a final simualtion results
#that take 20 percent sampes from teh first grup and 80 percent fropm teh enxty group to build a possible simualtion reuslts which will be use to 
#estimate the distrbutin of possible pucrhase in next 10 days.

import numpy as np
import os
import pandas
from datetime import datetime
os.chdir(r'C:\Users\mhosseini\Desktop\Python app')
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32   
from numba import cuda
import pyodbc


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
df['rplus']=np.round((df['totalpurchasebefore']+df['pramount'])/(df['dayfromstart']+df['time']+0.01))

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

#np.savetxt('x.csv',x,delimiter=",")
#
#data = np.loadtxt('test1.txt', dtype=int)
 

@cuda.jit
def Random_walk_Monte_carlo_simulation(data,k1,rng_states,time,gap,out):
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

import numba
@cuda.jit
def Markov_chain_Monte_carlo(data,k1,rng_states,time,transition_matrix,current_state,out):
    thread_id = cuda.grid(1)
    s=time
    future_states=cuda.local.array(shape=1000,dtype=numba.float64)#1000 is the maximum day forward we do the predcition
    #generate_states(transition_matrix,current_state,time,rng_states,features)
    for i in range(time):
        p=transition_matrix[current_state]
        if(current_state==0):
            if(xoroshiro128p_uniform_float32(rng_states, thread_id)<=p[0]):
                next_state = 0
            else:
                next_state = 1
        else:
            if(xoroshiro128p_uniform_float32(rng_states, thread_id)<=p[0]):
                next_state = 0
            else:
                next_state = 1
        future_states[i]=next_state
        current_state = next_state
    total=0
    x=0
    for i in range(time):
       x+= future_states[i]
    total=0
    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*k1.shape[0])
    nonzeroindex=np.int32(k1[randomidx,1]-1)
    if(x==0):
        total=0
    elif(nonzeroindex>time):
        total=-1
    else:
        if(nonzeroindex>=0):
            if (future_states[nonzeroindex]==1):
                total=total+k1[randomidx,3]
                s=k1[randomidx,7]
                f=k1[randomidx,6]+1
                x=x-1.0
                if ((k1[randomidx,3]==0) & (x > 0)):
                    total=-1.0
                elif((k1[randomidx,7]==0) & (k1[randomidx,6]==0) & (x>0.0)):
                     total=-1.0
                else:
                    while x>0 :
                        randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*(f-s)+s)
                        nonzeroindex=np.int32(nonzeroindex+data[randomidx,1])
                        if(nonzeroindex>time):
                            total=-1
                            break
                        elif (future_states[nonzeroindex]==1):
                             total=total+data[randomidx,3]
                             s=data[randomidx,7]
                             f=data[randomidx,6]+1
                             x=x-1.0
                             if((data[randomidx,7]==0) & (data[randomidx,6]==0) & (x>0.0)):
                                total=-1.0
                                break
                             elif((data[randomidx,3]==0) & (x>0.0)):
                                total=-1.0
                                break
                        else:
                            total=-1.0
                            break
            else:
                 total=-1.0
        else:
            total=-1.0
    out[thread_id]=complex(time,total)


import numba
@cuda.jit
def Markov_chain_Monte_carlo_v2(data,k1,rng_states,time,transition_matrix,current_state,out):
    thread_id = cuda.grid(1)
    future_states=cuda.local.array(shape=1000,dtype=numba.float64)#1000 is the maximum day forward we do the predcition
    #generate_states(transition_matrix,current_state,time,rng_states,features)
    for i in range(time):
        p=transition_matrix[current_state]
        if(current_state==0):
            if(xoroshiro128p_uniform_float32(rng_states, thread_id)<=p[0]):
                next_state = 0
            else:
                next_state = 1
        else:
            if(xoroshiro128p_uniform_float32(rng_states, thread_id)<=p[0]):
                next_state = 0
            else:
                next_state = 1
        future_states[i]=next_state
        current_state = next_state
    total=0
    x=0
    for i in range(time):
       x+= future_states[i]
    total=0
    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*k1.shape[0])
    nonzeroindex=np.int32(k1[randomidx,1]-1)
    if((x==0) & (nonzeroindex==-1)):
        total=0
    elif(nonzeroindex>=time):
        total=-1
    else:
        if(nonzeroindex>=0):
            if (future_states[nonzeroindex]==1):
                total=total+k1[randomidx,3]
                s=k1[randomidx,7]
                f=k1[randomidx,6]+1
                x=x-1.0
                if ((k1[randomidx,3]==0) & (x > 0)):
                    total=-1.0
                elif((k1[randomidx,7]==0) & (k1[randomidx,6]==0) & (x>0.0)):
                     total=-1.0
                else:
                    while x>0 :
                        randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*(f-s)+s)
                        nonzeroindex=np.int32(nonzeroindex+data[randomidx,1])
                        if(nonzeroindex>=time):
                            total=-1
                            break
                        elif (future_states[nonzeroindex]==1):
                             total=total+data[randomidx,3]
                             s=data[randomidx,7]
                             f=data[randomidx,6]+1
                             x=x-1.0
                             if((data[randomidx,7]==0) & (data[randomidx,6]==0) & (x>0.0)):
                                total=-1.0
                                break
                             elif((data[randomidx,3]==0) & (x>0.0)):
                                total=-1.0
                                break
                        else:
                            total=-1.0
                            break
            else:
                 total=-1.0
        else:
            total=-1.0
    out[thread_id]=complex(time,total)







def estimate_purchase(total_purchase,pramount,gap,ratio,timeforward,transition_matrix,current_state):
    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)) & (data[:,1]>=gap) & (data[:,4]>=2) & (data[:,4]<=4) & (data[:,1]<=timeforward))]
    #if(k1.shape[0]<15):
    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)) & (data[:,1]<=15))]
    if(k1.shape[0]>15):
        threads_per_block = 1024
        blocks = 2048
        seed=np.random.randint(1,100)
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed)
        Random_walk_out = np.zeros(threads_per_block * blocks, dtype=np.complex64)
        Markov_chain_out = np.zeros(threads_per_block * blocks, dtype=np.complex64)
        Random_walk_Monte_carlo_simulation[blocks, threads_per_block](data,k1,rng_states,timeforward,-gap,Random_walk_out)
        Markov_chain_Monte_carlo_v2[blocks, threads_per_block](data,k1,rng_states,timeforward,transition_matrix,current_state,Markov_chain_out)   
        x=np.imag(Random_walk_out)
        x=x[x>=0]
        y=np.imag(Markov_chain_out)
        y=y[y>=0]
        return(x,y)
        #return (np.quantile(x,[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
    else:
        return(-1,-1)
        #return (np.quantile(-1,[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
    


def grab_userid_data(userid,GamingServerID,datebreak,dyasbefore,daysafter):
    conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=PTSServer;'
                      'Database=backoffice;'
                      'Trusted_Connection=yes;')

    sql='''select a.UserID,a.GamingServerID,cast(prtime as date) as prtime,DATEDIFF(dd,cast(dateopened as date),cast(prtime as date)) as dayfromstart,sum(pramount) as pramount,
    DATEDIFF(dd,cast(prtime as date),'''+'\''+datebreak+'\''+''') as datediffs from backoffice..vw_UsersLookup a with(nolock) 
    join backoffice..sync_purchases b with(nolock) on a.UserID=b.userid and a.GamingServerID=b.gamingserverid
    where a.userid='''+str(userid)+'''
    and a.GamingServerID='''+str(GamingServerID)+''' and cast(prtime as date)<='''+'\''+datebreak+'\''+'''
    group by a.UserID,a.GamingServerID,cast(prtime as date),DATEDIFF(dd,cast(dateopened as date),cast(prtime as date))
    order by cast(prtime as date)'''
    datasource = pandas.read_sql(sql,conn)
    dayfromstart=np.sum(datasource['dayfromstart'])
    total_purchase=np.sum(datasource['pramount'])
    ratio=np.round(total_purchase/(dayfromstart+0.01))
    gap=(datetime.strptime(datebreak,"%Y-%m-%d") -  datetime.strptime(np.max(datasource['prtime']),"%Y-%m-%d")).days
    pramount=datasource.tail(1)['pramount'].item()
    history=np.zeros(dyasbefore)
    a=dyasbefore-datasource['datediffs']-1
    a=a[a>=0]
    history[a]=1
    history=history.astype(int)
    M = [[0]*2 for _ in range(2)]
    for (i,j) in zip(history,history[1:]):
        M[i][j] += 1
    for row in M:
        n = sum(row)
        if n > 0:
            row[:] = [f/sum(row) for f in row]
    transition_matrix = M
    transition_matrix=np.asarray(transition_matrix)
    current_state=history[-1]
    (x,y)=estimate_purchase(total_purchase,pramount,gap,ratio,daysafter,transition_matrix,current_state)
    return(x,y)
        
userid=28261765
GamingServerID=45
datebreak='2018-03-12'
dyasbefore=30
timeforward=15
#gap=(datetime.strptime(datebreak,"%Y-%m-%d") -  datetime.strptime(np.max(datasource['prtime']),"%Y-%m-%d")).days
(z,t)=grab_userid_data(userid,GamingServerID,datebreak,dyasbefore,timeforward)
result=[]
import random
for i in range(100000):
    if(random.uniform(0, 1)> 0.1):
        result.append(t[int(np.random.choice(t.shape[0],1))])
    else:
        result.append(z[int(np.random.choice(z.shape[0], 1))])

np.mean(result)
#total_purchase=df.iloc[i]['totalpurchasebefore']
#pramount=df.iloc[i]['pramount']
#time=df.iloc[i]['time to pass']
#ratio=df.iloc[i]['r']
#k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)) & (data[:,1]>=time) & (data[:,4]>=(0.9*ratio)) & (data[:,4]<=(1.1*ratio)))]
#  
    

#    sql='''select a.UserID,a.GamingServerID,cast(prtime as date) as prtime,DATEDIFF(dd,cast(dateopened as date),cast(prtime as date)) as dayfromstart,sum(pramount) as pramount,
#    DATEDIFF(dd,cast(prtime as date),'2018-09-20') as datediffs from backoffice..vw_UsersLookup a with(nolock) 
#    join backoffice..sync_purchases b with(nolock) on a.UserID=b.userid and a.GamingServerID=b.gamingserverid
#    where a.userid=4364635
#    and a.GamingServerID=474 and cast(prtime as date)<='20 Sep 2018'
#    group by a.UserID,a.GamingServerID,cast(prtime as date),DATEDIFF(dd,cast(dateopened as date),cast(prtime as date))
#    order by cast(prtime as date)'''





    
#df = pandas.read_csv('test.csv')
#for i in range(df.shape[0]):
#estimate_purchase(df.iloc[i]['totalpurchasebefore'],df.iloc[i]['pramount'],df.iloc[i]['time to pass'],df.iloc[i]['r'])
#
#
#weather_chain = MarkovChain(history=[0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1,1,0,1,1],states=[0,1])
#w=np.sum(weather_chain.generate_states(current_state=1, no=14))
#threads_per_block = 1024
#blocks = 1024
#seed=np.random.randint(1,100)
#rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed)
#out = np.zeros(threads_per_block * blocks, dtype=np.complex64)
#
#transition_matrix=np.asarray(transition_matrix)
#
#import csv
#with open("output.csv",'w') as resultFile:
#    wr = csv.writer(resultFile)
#    wr.writerows(w)
#
#df = pandas.read_csv('test.csv')
#df['r']=df['totalpurchasebefore']/(df['dayfromstart']+0.01)
#df['r']=np.round(df['r'])
#
#result=[]
#for i in range(df.shape[0]):
#    result.append(estimate_purchase(df.iloc[i]['totalpurchasebefore'],df.iloc[i]['pramount'],df.iloc[i]['time to pass'],df.iloc[i]['r']))
#    print(i)
#
#threads_per_block = 1024
#blocks = 512
#seed=np.random.randint(1,100)
#rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed)
#out = np.zeros(threads_per_block * blocks, dtype=np.complex64)
#
#conn = pyodbc.connect('Driver={SQL Server};'
#                      'Server=PTSServer;'
#                      'Database=backoffice;'
#                      'Trusted_Connection=yes;')
#
#sql='''select a.UserID,a.GamingServerID,cast(prtime as date) as prtime,DATEDIFF(dd,cast(dateopened as date),cast(prtime as date)) as dayfromstart,sum(pramount) as pramount,
#DATEDIFF(dd,cast(prtime as date),'2018-09-20') as datediffs from backoffice..vw_UsersLookup a with(nolock) 
#join backoffice..sync_purchases b with(nolock) on a.UserID=b.userid and a.GamingServerID=b.gamingserverid
#where a.userid=4364635
# and a.GamingServerID=474 and cast(prtime as date)<='20 Sep 2018'
#group by a.UserID,a.GamingServerID,cast(prtime as date),DATEDIFF(dd,cast(dateopened as date),cast(prtime as date))
#order by cast(prtime as date)'''
#
#data = pandas.read_sql(sql,conn)
#
#dayfromstart=np.sum(data['dayfromstart'])
#totalpurchasebefore=np.sum(data['pramount'])
#ratio=np.round(totalpurchasebefore/(totalpurchasebefore+0.01))
#gap=datetime.strptime('2018-09-20',"%Y-%m-%d") -  datetime.strptime(np.max(data['prtime']),"%Y-%m-%d") 
#time=dayforward
#pramount=data.tail(1)['pramount']
#history=np.zeros(dyasbefore)
#a=dyasbefore-data['datediffs']-1
#a=a[a>=0]
#history[a]=1
#M = [[0]*2 for _ in range(2)]
#for (i,j) in zip(history,history[1:]):
#    M[i][j] += 1
#for row in M:
#    n = sum(row)
#    if n > 0:
#        row[:] = [f/sum(row) for f in row]
#transition_matrix = M
#transition_matrix=np.asarray(transition_matrix)
#current_state=history[-1]
q=[]
for ww in range(2351):
    shit=[]
    total_purchase=5252.0
    pramount=307.0
    ratio=3.0
    dayfromstart=1737
    future_states=[]#1000 is the maximum day forward we do the predcition
    #generate_states(transition_matrix,current_state,time,rng_states,features)
    for i in range(15):
        p=transition_matrix[current_state]
        if(current_state==0):
            if(random.uniform(0,1)<=p[0]):
                next_state = 0
            else:
                next_state = 1
        else:
            if(random.uniform(0,1)<=p[0]):
                next_state = 0
            else:
                next_state = 1
        future_states.append(next_state)
        current_state = next_state
    index=(np.nonzero(future_states))[0]
    a=np.diff(index)
    index=index+1
    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)) & (data[:,1]==index[0]) & (data[:,4]>=(0.9*ratio)) & (data[:,4]<=(1.1*ratio)))]
    if(k1.shape[0]>0):
        b=int(np.random.randint(k1.shape[0], size=1))
        shit.append(b)
        total_purchase=k1[b,2]+k1[b, 3]+k1[b,0]
        pramount=k1[b, 3]
        dayfromstart=dayfromstart+index[0]
        ratio=total_purchase/(dayfromstart)
        for i in range(len(a)):
            k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)) & (data[:,1]==a[i]) & (data[:,4]>=(0.9*ratio)) & (data[:,4]<=(1.1*ratio)))]
            if(k1.shape[0]>0):
                b=int(np.random.randint(k1.shape[0], size=1))
                shit.append(b)
                total_purchase=total_purchase+k1[b, 3]
                pramount=k1[b, 3]
                dayfromstart=dayfromstart+a[i]
                ratio=total_purchase/(dayfromstart)
            else:
                total_purchase=-1
                break
        if(total_purchase >0):
            print(total_purchase,shit)
        q.append(total_purchase)
        
        

    
    
    
    
    
    




