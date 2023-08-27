import numpy as np
import os
import pandas
from datetime import datetime
os.chdir(r'C:\Users\mhosseini\Desktop\Python app')
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32   
from numba import cuda
import pyodbc
import numba
import random
import swifter

import scipy

df = pandas.read_csv('userid pramount paamount sice june 2018.csv')
del df['UserID']
del df['GamingServerID']
df=df.loc[(df.totalpurchasebefore>= 50)]
df=df.sort_values(['pramount','totalpurchasebefore','dayfromstart'], ascending=[True, True,True])
x=range(0,df.shape[0])
df['indexx']=x
x=np.zeros(df.shape[0])
df['maxx']=x
df['minn']=x
df['totalpluspr']=df['totalpurchasebefore']+df['pramount']
df['dayfromstartplus']=df['dayfromstart']+df['time']

data=df.values

from numba import guvectorize, cuda,njit, float64, void,prange


#@numba.jit(parallel=True,nopython=True)
#def cVestDiscount (data):
#    out = np.empty((data.shape[0],600), dtype=np.int32)
#    for i in prange(0,data.shape[0]):
#        k1=data[np.where((data[:,0]>=float(0.9*data[i,4])) & (data[:,0]<=float(1.1*data[i,4])) & (data[:,3]>=float(0.9*data[i,9])) & (data[:,3]<=float(1.1*data[i,9])) & (data[:,5]>=float(0.9*data[i,10])) & (data[:,5]<=float(1.1*data[i,10])))]
#        for j in range(k1[:,6].shape[0]):
#            out[i,j]=k1[j,6]
#    return out


@numba.jit()
def cVestDiscount_nonparallel(data):
    out = []
    for i in range(0,data.shape[0]):
        k1=data[np.where((data[:,0]>=float(0.9*data[i,4])) & (data[:,0]<=float(1.1*data[i,4])) & (data[:,3]>=float(0.9*data[i,9])) & (data[:,3]<=float(1.1*data[i,9])) & (data[:,5]>=float(0.9*data[i,10])) & (data[:,5]<=float(1.1*data[i,10])))]
        out.append(k1[:,6])
    return out



@cuda.jit
def Random_walk_Monte_carlo_simulation_with_index(data,out,ind,k1,rng_states,time,gap,output):
    t=gap
    thread_id = cuda.grid(1)
    total=0
    future_states=cuda.local.array(shape=1000,dtype=numba.float64)
    for i in range(time):
        future_states[i]=0
    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*k1.shape[0])
    if((t+k1[randomidx,1])<=time):
            index=int(k1[randomidx,1]+t-1)#gap is negative
            t=t+k1[randomidx,1]
            future_states[index]=1
            total=total+k1[randomidx,4]
            indexy=np.int(k1[randomidx,6])
            if (k1[randomidx,4]==0):
                output[thread_id]=complex(0,total)
            elif(ind[indexy]==-1):
                output[thread_id]=complex(0,-1.0)
            else:
                while t<=time: #t<=time
                    z=np.int(ind[indexy])
                    nonzerolength=0
                    for i in range(1,(len(out[z])-1)):
                        if (out[z][i]==0) & (out[z][i-1]==0):
                            break
                        else:
                            nonzerolength=nonzerolength+1
                    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*nonzerolength)
                    currentindex=np.int(out[z][randomidx])
                    if((t+data[currentindex,1])<=time):
                         t=t+data[currentindex,1]
                         index=int(index+data[currentindex,1])
                         future_states[index]=1
                         total=total+data[currentindex,4]
                         indexy=currentindex
                         if(data[currentindex,4]==0):
                             break
                         elif(ind[indexy]==-1):
                            total=-1.0
                            break
                    else:
                        break
                x=0
                for i in range(time):
                    x=x+(future_states[time-i-1])*(2**i)
#                y=0
#                for i in range(time):
#                    y=y+future_states[i]
                output[thread_id]=complex(x,total)
    else:
        x=0
        for i in range(time):
            x=x+(future_states[time-i-1])*(2**i)
        output[thread_id]=complex(x,total) 
        

#@cuda.jit
#def Random_walk_Monte_carlo_simulation_with_index(data,out,ind,k1,rng_states,time,gap,output):
#    t=gap
#    thread_id = cuda.grid(1)
#    total=0
#    future_states=numba.cuda.local.array(shape=1000,dtype=numba.float64)
#    for i in range(time):
#        future_states[i]=0
#    randomidx=0
#    index=int(k1[randomidx,1]+t-1)#gap is negative
#    t=t+k1[randomidx,1]
#    future_states[index]=1
#    total=total+k1[randomidx,4]
#    indexy=816228
#    while t<=time: 
#        z=np.int(ind[indexy])
#        if(z>=0):
#            nonzerolength=0
#            for i in range(1,(len(out[z])-1)):
#                if (out[z][i]==0) & (out[z][i-1]==0):
#                    break
#                else:
#                    nonzerolength=nonzerolength+1
#            randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*nonzerolength)
#            currentindex=np.int(out[z][randomidx])
#            if(data[currentindex,1]==0):
#                break
#            elif((t+data[currentindex,1])<=time):
#                 t=t+data[currentindex,1]
#                 index=int(index+data[currentindex,1])
#                 future_states[index]=1
#                 total=total+data[currentindex,4]
#                 indexy=currentindex
#            else:
#                break
#        else:
#            total=-1.0
#            break
#    if total==-1.0:
#        output[thread_id]=complex(-1,-1)
#    else: 
##        for i in range(index+1,time):
##            future_states[i]=0
#        x=0
#        for i in range(0,time):
#            if(future_states[(time-i-1)]==1):
#                x+=(2**i)
#        x=float64(x)
#        y=67108864.0-x
##        for i in range(0,time):
##            if (future_states[i]==1):
##                y=i
##            #y=len(future_states)#+=(1-future_states[i])
#        output[thread_id]=complex(x,y)



def estimate_purchase(total_purchase,pramount,gap,dayfromstart,timeforward,transition_matrix,current_state,history,dyasbefore):
    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) &\
                     (data[:,3]<=(1.1*total_purchase)) & (data[:,1]>gap) & (data[:,1]<=(gap+timeforward)) & (data[:,5]>=(0.9*dayfromstart)) &\
                     (data[:,5]<=(1.1*dayfromstart)))]
    k2=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) &\
                     (data[:,3]<=(1.1*total_purchase)) & (data[:,1]==0) & (data[:,5]>=(0.9*dayfromstart)) &\
                     (data[:,5]<=(1.1*dayfromstart)))]
    k1=np.vstack((k1,k2))
#    if(k1.shape[0]<15):
#        k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)))]
    if(k1.shape[0]>15):
        threads_per_block = 512
        blocks = 512
        seed=np.random.randint(1,100)
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed)
        Random_walk_out = np.zeros(threads_per_block * blocks, dtype=np.complex128)
        Random_walk_Monte_carlo_simulation_with_index[blocks, threads_per_block](data,out,ind,k1,rng_states,timeforward,-gap,Random_walk_out)
        purchase=np.imag(Random_walk_out)
        pattern=np.real(Random_walk_out)
        d=pandas.DataFrame({'purchase':purchase,'pattern':pattern})
        d=d[d.purchase>=0]
        d.loc[d.pattern<0,'pattern']=0
        #d['patternb']="{0:b}".format(d.pattern)
        d['patternb']=d.pattern.apply(lambda x:"{0:b}".format(int(x)))
        d['patternb']=d.patternb.apply(lambda x:str(x).zfill(timeforward))
        x=0
        for i in range(len(history)):
            x=x+(history[len(history)-i-1])*(2**i)
        d.loc[:,'history']=x
        d['bhistory']=d.history.apply(lambda x:"{0:b}".format(int(x)))
        d['bhistory']=d.bhistory.apply(lambda x:str(x).zfill(dyasbefore))
        return(d)
        #return (np.quantile(x,[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
    else:
        return(-1)
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
    dayfromstart=np.max(datasource['dayfromstart'])
    total_purchase=np.sum(datasource['pramount'])
    #ratio=np.round(total_purchase/(dayfromstart+0.01))
    gap=(datetime.strptime(datebreak,"%Y-%m-%d") -  datetime.strptime(np.max(datasource['prtime']),"%Y-%m-%d")).days
    history_length=(datetime.strptime(datebreak,"%Y-%m-%d") -  datetime.strptime(np.min(datasource['prtime']),"%Y-%m-%d")).days+1
    pramount=datasource.tail(1)['pramount'].item()
    
    if history_length<dyasbefore:
        dyasbefore=history_length  
        
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
    d=estimate_purchase(total_purchase,pramount,gap,dayfromstart,daysafter,transition_matrix,current_state,history,dyasbefore)
    return(d)
        




out=cVestDiscount_nonparallel(data)
length = max(map(len, out))
y=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in out])    
s=out
out=y

ind=np.zeros(out.shape[0])
ind=ind-1
counter=0
for i in range(0,out.shape[0]):
    if len(np.nonzero(out[i])[0])>0:
        ind[i]=counter
        counter=counter+1
out=out[~np.all(out == 0,axis=1)]

out=np.ascontiguousarray(out)
k1=np.ascontiguousarray(k1)
ind=np.ascontiguousarray(ind)
data=np.ascontiguousarray(data)





#userid=4189425
#GamingServerID=474
#datebreak='2018-03-12'
#dyasbefore=20
#timeforward=30
#test=grab_userid_data(userid,GamingServerID,datebreak,dyasbefore,timeforward)
#
#
#
#p_value=[]
#for i in range(test.shape[0]):
#    n1=IntVector(test.iloc[i]['patternb'])
#    n2=IntVector(test.iloc[i]['bhistory'])
#    p_value.append(r_f(n1,n2)[0])
#
#test['pvalue']=p_value
#
#
#
#
sample = pandas.read_csv('sim.csv')
quantile=[]
for j in range(100): #i in range(sample.shape[0])
    userid=sample.iloc[j]['userid']
    GamingServerID=sample.iloc[j]['gamingserverid']
    datebreak='2018-03-12'
    dyasbefore=20
    timeforward=30
    test=grab_userid_data(userid,GamingServerID,datebreak,dyasbefore,timeforward)
    if(type(test)!=int):
        if(len(test['bhistory'][0])>=20):
            p_value=[]
            for i in range(test.shape[0]):
                n1=IntVector(test.iloc[i]['patternb'])
                n2=IntVector(test.iloc[i]['bhistory'])
                p_value.append(r_f(n1,n2)[0])
            test['pvalue']=p_value
            homogen=test.loc[(test.pvalue > 0.1) , ['purchase']]
            non_homogen=test.loc[(test.pvalue <0.1), ['purchase']]
            
            final_purchase_value=[]
            for i in range(10000):
                if random.uniform(0, 1)<0.1675465:
                    final_purchase_value.append(non_homogen.sample(1)['purchase'].iloc[0])
                else:
                    final_purchase_value.append(homogen.sample(1)['purchase'].iloc[0])
            
            #quantile.append(np.quantile(final_purchase_value,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
            quantile.append(final_purchase_value)
        else:
            quantile.append([-1])
    else:
        quantile.append([-1])
    print(j)
    
    
length = max(map(len, quantile))
y=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in quantile])


#test.to_csv('pattern_sim.csv', sep=',')



#calling r pckage find homgenioty
import rpy2.robjects.packages as rpackages

utils = rpackages.importr('utils')

from rpy2.robjects import IntVector,r
from rpy2 import robjects

n1=IntVector([0,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,1,0])
n2=IntVector([0,1,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,1,1])

robjects.r('''
        # create a function `f`
        library('markovchain')
        f <- function(n1,n2, verbose=FALSE) {
            verifyHomogeneity(list(n1,n2),verbose = FALSE)$pvalue
        }
        # call the function `f` with argument value 3
        ''')
r_f = robjects.globalenv['f']

res=r_f(n1,n2)[0]










#from random import randint
#
#x=[]
#for j in range(10000):
#    pramount=307.0
#    total_purchase=5252.0
#    dayfromstart=71
#    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) & (data[:,3]<=(1.1*total_purchase)) & (data[:,1]<=timeforward) & (data[:,5]>=(0.9*dayfromstart)) & (data[:,5]<=(1.1*dayfromstart)))]
#    total=0
#    time=0
#    for i in range(0,10000):
#        if(k1.shape[0]>0):
#            b=randint(0, k1.shape[0]-1)
#            if (time+k1[b][1])<=timeforward:
#                total=total+k1[b][4]
#                dayfromstart=k1[b][5]+k1[b][1]
#                pramount=k1[b][4]
#                time=time+k1[b][1]
#                total_purchase=k1[b][3]+k1[b][0]
#                k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) & (data[:,3]<=(1.1*total_purchase)) & (data[:,5]>=(0.9*dayfromstart)) & (data[:,5]<=(1.1*dayfromstart)))]
#            else:
#                break
#        else:
#            break
#    x.append(total)
#    print(j)






#out=cVestDiscount(data[0:4997559,])

#np.savetxt('test1.txt', out, fmt='%d')
#out = np.loadtxt('test1.txt', dtype=int) 
#
#
#
#nonzeroout=[]
#for i in range(0,out.shape[0]):
#    nonzeroout.append(out[i][np.nonzero(out[i])])



#@cuda.jit
#def Random_walk_Monte_carlo_simulation_2(data,out,ind,k1,rng_states,time,gap,output):
#    t=gap
#    thread_id = cuda.grid(1)
#    total=0
##    future_states=cuda.local.array(shape=1000,dtype=numba.float64)
##    for i in range(time):
##        future_states[i]=0
##    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*k1.shape[0])
##    if((t+k1[randomidx,1])<=time):
##            t=t+k1[randomidx,1]
##            index=int(k1[randomidx,1])
##            future_states[index]=1
##            total=total+k1[randomidx,4]
##            indexy=np.int(k1[randomidx,6])
##            if (k1[randomidx,4]==0):
##                out[thread_id]=complex(t,total)
##            elif(len(outnonzero[indexy])==0):
##                out[thread_id]=complex(t,-1.0)
##            else:
##                while t<=time: #t<=time
##                    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*len(outnonzero[indexy]))
##                    currentindex=np.int(outnonzero[indexy][randomidx])
##                    if((t+data[currentindex,1])<=time):
##                         t=t+data[currentindex,1]
##                         index=int(index+data[currentindex,1])
##                         future_states[index]=1
##                         total=total+data[currentindex,3]
##                         indexy=currentindex
##                         if(data[currentindex,4]==0):
##                             break
##                         elif(len(outnonzero[indexy])==0):
##                            total=-1.0
##                            break
##                    else:
##                        break
##                x=0
##                for i in range(time):
##                    x=x+(future_states[time-i-1])*(2**i)
##                y=0
##                for i in range(time):
##                    y=y+future_states[i]
##                out[thread_id]=complex(x,total)
##    else:
##        x=0
##        for i in range(time):
##            x=x+(future_states[time-i-1])*(2**i)
##        y=0
##        for i in range(time):
##            y=y+future_states[i]
##        out[thread_id]=complex(x,total)  
#    output[thread_id]=complex(1,1)  


                
#@guvectorize(['void(float64[:], float64[:,:],float64[:,:],float64[:])'], '(o),(m,o),(s,t)-> (t)', target='cuda', nopython=True)
#def cVestDiscount (datax, data,x,out):
#        for ID in range(0,data.shape[0]):
#            if ((data[ID,0]>=float(0.9*datax[4])) & (data[ID,0]<=float(1.1*datax[4])) &\
#                (data[ID,3]>=float(0.9*datax[9])) & (data[ID,3]<=float(1.1*datax[9])) &\
#                (data[ID,5]>=float(0.9*datax[10])) & (data[ID,5]<=float(1.1*datax[10]))):
#                out[ID]=ID
#
#               
#out = np.zeros(datax.shape[0]*200, dtype=np.float64).reshape(datax.shape[0], 200) 
# 
#x= np.zeros(datax.shape[0]*200, dtype=np.float64).reshape(datax.shape[0], 200) 
#
#       
#data=np.ascontiguousarray(data, dtype=np.float64)
#out=np.ascontiguousarray(out, dtype=np.float64)
#x=np.ascontiguousarray(x, dtype=np.float64)
#datax=np.ascontiguousarray(datax, dtype=np.float64)
#
#out = cVestDiscount(datax, data,x,out)

#from numba import njit, jit, guvectorize
#import math
#
#@guvectorize(["void(float64[:], float64[:])"], "(n) -> ()", target="cuda", nopython=True)
#def row_sum_gu(input, output) :
#    a = 0.
#    for i in range(input.size):
#        a = math.sqrt(a**2 + input[i]**2)
#    output[0] =a
#
#rows = int(64)
#columns = int(1e6)
#
#input_array = np.random.random((rows, columns))
#output_array = np.zeros((rows))
#
#output_array=row_sum_gu(input_array, output_array)
#
#
#
#
#
#out = np.zeros(data.shape[0]*1000, dtype=np.float32).reshape(data.shape[0], 1000)
#
#@guvectorize(['void(float32[:], float32[:])'], '(n) ->(n)', target='cuda')
#def my_func(inp, out):
#    tmp1 = 0.
#    tmp = inp[0]
#    for i in range(out.shape[0]):
#        tmp1 += tmp
#        out[i] = tmp1
#        tmp *= inp[0]
#            
#dev_inp = cuda.to_device(inp)             # alloc and copy input data
#
#my_func(dev_inp, dev_inp)             # invoke the gufunc
#
#dev_inp.copy_to_host(inp)    
#
#
#
#import numpy as np
#from numba import guvectorize
#import time
#from timeit import default_timer as timer
#
#
#@guvectorize(['void(int64, float64[:,:], float64[:,:,:], int64, int64, float64[:,:,:])'], '(),(m,o),(n,m,o),(),() -> (n,m,o)', target='cuda', nopython=True)
#def cVestDiscount (countRow, multBy, discount, n, countCol, cv):
#    for as_of_date in range(0,countRow):
#        for ID in range(0,countCol):
#            for num in range(0,n):
#                cv[as_of_date][ID][num] = multBy[ID][num] * discount[as_of_date][ID][num]
#
#countRow = np.int64(100)
#multBy = np.float64(np.arange(20000).reshape(4000,5))
#discount = np.float64(np.arange(2000000).reshape(100,4000,5))
#n = np.int64(5)
#countCol = np.int64(4000)
#cv = np.zeros(shape=(100,4000,5), dtype=np.float64)
#func_start = timer()
#cv = cVestDiscount(countRow, multBy, discount, n, countCol, cv)
#timing=timer()-func_start
#print("Function: discount factor cumVest duration (seconds):" + str(timing))
#
#import numpy as np
#from numba import guvectorize
#from timeit import default_timer as timer
#
#
#@guvectorize(['void(float64[:,:], float64[:,:], int64, int64, float64[:,:])'], '(m,o),(m,o),(),() -> (m,o)', target='cuda', nopython=True)
#def cVestDiscount (multBy, discount, n, countCol, cv):
#        for ID in range(0,countCol):
#            for num in range(0,n):
#                cv[ID][num] = multBy[ID][num] * discount[ID][num]
#
#multBy = np.float64(np.arange(20000).reshape(4000,5))
#discount = np.float64(np.arange(2000000).reshape(100,4000,5))
#n = np.int64(5)
#countCol = np.int64(4000)
#cv = np.zeros(shape=(100,4000,5), dtype=np.float64)
#cv = cVestDiscount(multBy, discount, n, countCol, cv)
#timing=timer()-func_start
#print("Function: discount factor cumVest duration (seconds):" + str(timing))


df = pandas.read_csv('userid_pramount_time_pramountmoneybringforward_jan_2018_han_2019 country.csv')
del df['UserID']
del df['GamingServerID']


df=df.loc[(df.totalpurchasebefore>= 50)]
df=df.sort_values(['pramount','totalpurchasebefore','dayfromstart'], ascending=[True, True,True])
df['ratio']=df['totalpurchasebefore']/df['dayfromstart']



df=df[['pramount','time','purchasecount','totalpurchasebefore','pramount2','dayfromstart','ratio','country']]

data=df.values



sample = pandas.read_csv('sim.csv')
quantile=[]
for j in range(sample.shape[0]): #i in range(sample.shape[0])
    userid=sample.iloc[j]['userid']
    GamingServerID=sample.iloc[j]['gamingserverid']
    datebreak='2018-03-12'
    dyasbefore=20
    timeforward=30
    
    conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=PTSServer;'
                      'Database=backoffice;'
                      'Trusted_Connection=yes;')

    sql='''select a.UserID,a.GamingServerID,country,cast(prtime as date) as prtime,DATEDIFF(dd,cast(dateopened as date),cast(prtime as date)) as dayfromstart,sum(pramount) as pramount,
    DATEDIFF(dd,cast(prtime as date),'''+'\''+datebreak+'\''+''') as datediffs from backoffice..vw_UsersLookup a with(nolock) 
    join backoffice..sync_purchases b with(nolock) on a.UserID=b.userid and a.GamingServerID=b.gamingserverid
    where a.userid='''+str(userid)+'''
    and a.GamingServerID='''+str(GamingServerID)+''' and cast(prtime as date)<='''+'\''+datebreak+'\''+'''
    group by a.UserID,a.GamingServerID,country,cast(prtime as date),DATEDIFF(dd,cast(dateopened as date),cast(prtime as date))
    order by cast(prtime as date)'''
    datasource = pandas.read_sql(sql,conn)
    dayfromstart=np.max(datasource['dayfromstart'])
    total_purchase=np.sum(datasource[:-1]['pramount'])
    ratio=np.round(total_purchase/(dayfromstart+0.01))
    country=datasource.iloc[0]['country']
    gap=(datetime.strptime(datebreak,"%Y-%m-%d") -  datetime.strptime(np.max(datasource['prtime']),"%Y-%m-%d")).days
    history_length=(datetime.strptime(datebreak,"%Y-%m-%d") -  datetime.strptime(np.min(datasource['prtime']),"%Y-%m-%d")).days+1
    pramount=datasource.tail(1)['pramount'].item()
    
    if history_length<dyasbefore:
        dyasbefore=history_length  
        
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
    

    purchase=[]
    pattern=[]
    for k in range(200000):
        t=-gap
        k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) &\
                         (data[:,3]<=(1.1*total_purchase)) & (data[:,6]>=(0.9*ratio)) &\
                         (data[:,6]<=(1.1*ratio)) &(data[:,7]==country))]
        total=0
        time=0
        future_states=np.zeros(30)
        index=t-1
        bonus=0
        for i in range(0,10000):
            if(k1.shape[0]>0):
                b=randint(0, k1.shape[0]-1)
                index=int(index+k1[b,1])
                if index >= 0:
                    if (time+k1[b][1])<=timeforward:
                        future_states[index]=1
                        total=total+k1[b][4]
                        dayfromstart=k1[b][5]+k1[b][1]
                        pramount=k1[b][4]
                        time=time+k1[b][1]
                        total_purchase=k1[b][3]+k1[b][0]
                        ratio=total_purchase/dayfromstart
                        k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) &\
                         (data[:,3]<=(1.1*total_purchase)) & (data[:,6]>=(0.9*ratio)) &\
                         (data[:,6]<=(1.1*ratio)) &(data[:,7]==country))]
                    else:
                        break
                else:
                    break
            else:
                total=-1
                break
        purchase.append(total)
        c=0
        for i in range(30):
            c=c+(future_states[30-i-1])*(2**i)
        pattern.append(c)
        print(k)
        
        
    test=pandas.DataFrame({'purchase':purchase,'pattern':pattern})
    test=test[test.purchase>=0]
    if(test.shape[0]>0):
        test['patternb']=test.pattern.apply(lambda x:"{0:b}".format(int(x)))
        test['patternb']=test.patternb.apply(lambda x:str(x).zfill(timeforward))
        x=0
        for i in range(len(history)):
            x=x+(history[len(history)-i-1])*(2**i)
        test.loc[:,'history']=x
        test['bhistory']=test.history.apply(lambda x:"{0:b}".format(int(x)))
        test['bhistory']=test.bhistory.apply(lambda x:str(x).zfill(dyasbefore))
        if(type(test)!=int):
            if((len(test['bhistory'][0])>=20)):
                p_value=[]
                for i in range(test.shape[0]):
                    n1=IntVector(test.iloc[i]['patternb'])
                    n2=IntVector(test.iloc[i]['bhistory'])
                    p_value.append(r_f(n1,n2)[0])
                test['pvalue']=p_value
                homogen=test.loc[(test.pvalue > 0.05) , ['purchase']]
                non_homogen=test.loc[(test.pvalue <0.05), ['purchase']]
                
                final_purchase_value=[]
                for i in range(1000):
                    if random.uniform(0, 1)<0.1435425:
                        final_purchase_value.append(non_homogen.sample(1)['purchase'].iloc[0])
                    else:
                        final_purchase_value.append(homogen.sample(1)['purchase'].iloc[0])
                
                #quantile.append(np.quantile(final_purchase_value,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
                quantile.append(final_purchase_value)
            else:
                quantile.append([-1])
        else:
            quantile.append([-1])
    else:
        quantile.append([-1])
    print(j)

#purchase=[]
#pattern=[]
#Bonus=[]
#for j in range(2000):
#    pramount=30
#    total_purchase=440
#    t=-1
#    ratio=18.0
#    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) &\
#                     (data[:,3]<=(1.1*total_purchase)) & (data[:,6]>=(0.9*ratio)) &\
#                     (data[:,6]<=(1.1*ratio)) &(data[:,7]=='United Kingdom'))]
#    total=0
#    time=0
#    future_states=np.zeros(30)
#    index=t-1
#    bonus=0
#    for i in range(0,10000):
#        if(k1.shape[0]>0):
#            b=randint(0, k1.shape[0]-1)
#            index=int(index+k1[b,1])
#            if index >= 0:
#                if (time+k1[b][1])<=timeforward:
#                    future_states[index]=1
#                    total=total+k1[b][4]
#                    dayfromstart=k1[b][5]+k1[b][1]
#                    pramount=k1[b][4]
#                    time=time+k1[b][1]
#                    total_purchase=k1[b][3]+k1[b][0]
#                    ratio=total_purchase/dayfromstart
#                    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) &\
#                     (data[:,3]<=(1.1*total_purchase)) & (data[:,6]>=(0.9*ratio)) &\
#                     (data[:,6]<=(1.1*ratio)) &(data[:,7]=='United Kingdom'))]
#                else:
#                    break
#            else:
#                break
#        else:
#            total=-1
#            break
#    purchase.append(total)
#    c=0
#    for i in range(30):
#        c=c+(future_states[30-i-1])*(2**i)
#    pattern.append(c)
#    Bonus.append(bonus)
#    print(j)
#   
#d=pandas.DataFrame({'purchase':purchase,'pattern':pattern,'bonus':Bonus})
#d=d[d.purchase>=0]
#d['patternb']=d.pattern.apply(lambda x:"{0:b}".format(int(x)))
#d['patternb']=d.patternb.apply(lambda x:str(x).zfill(timeforward))
#x=0
#for i in range(len(history)):
#    x=x+(history[len(history)-i-1])*(2**i)
#d.loc[:,'history']=x
#d['bhistory']=d.history.apply(lambda x:"{0:b}".format(int(x)))
#d['bhistory']=d.bhistory.apply(lambda x:str(x).zfill(dyasbefore))
#
#
#p_value=[]
#for i in range(d.shape[0]):
#    n1=IntVector(d.iloc[i]['patternb'])
#    n2=IntVector(d.iloc[i]['bhistory'])
#    p_value.append(r_f(n1,n2)[0])
#d['pvalue']=p_value
#homogen=d.loc[(d.pvalue > 0.05) , ['purchase','bonus']]
#non_homogen=d.loc[(d.pvalue <=0.05), ['purchase','bonus']]
#
#final_purchase_value=[]
#final_bonus_value=[]
#for i in range(1000):
#    if random.uniform(0, 1)<0.1435425:
#        x=randint(0,non_homogen.shape[0]-1)
#        final_purchase_value.append(non_homogen.iloc[x]['purchase'])
#        final_bonus_value.append(non_homogen.iloc[x]['bonus'])
#    else:
#        x=randint(0,homogen.shape[0]-1)
#        final_purchase_value.append(homogen.iloc[x]['purchase'])
#        final_bonus_value.append(homogen.iloc[x]['bonus'])
#
#np.quantile(final_purchase_value,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])




#all the paralle inclusing ratio

userid_pramount_time_pramount_2_bonus_march_2018_han_2019
userid pramount paamount sice june 2018

import numpy as np
import os
import pandas
from datetime import datetime
os.chdir(r'C:\Users\mhosseini\Desktop\Python app')
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32   
from numba import cuda
import pyodbc
import numba
import random
import swifter

import scipy

df = pandas.read_csv('userid pramount paamount sice june 2018.csv')
del df['UserID']
del df['GamingServerID']
#del df['bonus']
df=df.loc[(df.totalpurchasebefore>= 50)]
df=df.sort_values(['pramount','totalpurchasebefore','dayfromstart'], ascending=[True, True,True])
x=range(0,df.shape[0])
df['indexx']=x
x=np.zeros(df.shape[0])
df['maxx']=x
df['minn']=x
df['totalpluspr']=df['totalpurchasebefore']+df['pramount']
df['dayfromstartplus']=df['dayfromstart']+df['time']
df['ratio']=df['totalpurchasebefore']/df['dayfromstart']
df['ratioplus']=df['totalpluspr']/df['dayfromstartplus']

data=df.values

from numba import guvectorize, cuda,njit, float64, void,prange



@numba.jit()
def cVestDiscount_nonparallel(data):
    out = []
    for i in range(0,data.shape[0]):
        k1=data[np.where((data[:,0]>=float(0.85*data[i,4])) & (data[:,0]<=float(1.15*data[i,4])) & (data[:,3]>=float(0.85*data[i,9])) &\
                         (data[:,3]<=float(1.15*data[i,9])) & (data[:,5]>=float(0.85*data[i,10])) & (data[:,5]<=float(1.15*data[i,10])) &\
                         (data[:,2]==data[i,2]+1))]
        out.append(k1[:,6])
    return out


                                                    
@cuda.jit
def Random_walk_Monte_carlo_simulation_with_index(data,out,ind,k1,rng_states,time,gap,output):
    t=gap
    thread_id = cuda.grid(1)
    total=0
    future_states=cuda.local.array(shape=1000,dtype=numba.float64)
    for i in range(time):
        future_states[i]=0
    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*k1.shape[0])
    if((t+k1[randomidx,1])<=time):
            index=int(k1[randomidx,1]+t-1)#gap is negative
            t=t+k1[randomidx,1]
            future_states[index]=1
            total=total+k1[randomidx,4]
            indexy=np.int(k1[randomidx,6])
            if (k1[randomidx,4]==0):
                output[thread_id]=complex(0,total)
            elif(ind[indexy]==-1):
                output[thread_id]=complex(0,-1.0)
            else:
                while t<=time: #t<=time
                    z=np.int(ind[indexy])
                    nonzerolength=0
                    for i in range(1,(len(out[z])-1)):
                        if (out[z][i]==0) & (out[z][i-1]==0):
                            break
                        else:
                            nonzerolength=nonzerolength+1
                    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*nonzerolength)
                    currentindex=np.int(out[z][randomidx])
                    if((t+data[currentindex,1])<=time):
                         t=t+data[currentindex,1]
                         index=int(index+data[currentindex,1])
                         future_states[index]=1
                         total=total+data[currentindex,4]
                         indexy=currentindex
                         if(data[currentindex,4]==0):
                             break
                         elif(ind[indexy]==-1):
                            total=-1.0
                            break
                    else:
                        break
                x=0
                for i in range(time):
                    x=x+(future_states[time-i-1])*(2**i)
#                y=0
#                for i in range(time):
#                    y=y+future_states[i]
                output[thread_id]=complex(x,total)
    else:
        x=0
        for i in range(time):
            x=x+(future_states[time-i-1])*(2**i)
        output[thread_id]=complex(x,total) 
       
def estimate_purchase(total_purchase,pramount,gap,dayfromstart,timeforward,transition_matrix,current_state,history,ratio,dyasbefore):
#    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) &\
#                     (data[:,3]<=(1.1*total_purchase)) & (data[:,1]>gap) & (data[:,1]<=(gap+timeforward)) & (data[:,11]>=(0.9*ratio)) &\
#                     (data[:,11]<=(1.1*ratio)))]
#    k2=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) &\
#                     (data[:,3]<=(1.1*total_purchase)) & (data[:,1]==0) & (data[:,11]>=(0.9*ratio)) &\
#                     (data[:,11]<=(1.1*ratio)))]
#    k1=np.vstack((k1,k2))
    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) &\
                     (data[:,3]<=(1.1*total_purchase)) & (data[:,11]>=(0.9*ratio)) &\
                     (data[:,11]<=(1.1*ratio)))]
#    k2=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,3]>=(0.9*total_purchase)) &\
#                     (data[:,3]<=(1.1*total_purchase)) & (data[:,1]==0) & (data[:,11]>=(0.9*ratio)) &\
#                     (data[:,11]<=(1.1*ratio)))]
#    k1=np.vstack((k1,k2))
    
    
#    if(k1.shape[0]<15):
#        k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)))]
    if(k1.shape[0]>15):
        threads_per_block = 512
        blocks = 512
        seed=np.random.randint(1,100)
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed)
        Random_walk_out = np.zeros(threads_per_block * blocks, dtype=np.complex128)
        Random_walk_Monte_carlo_simulation_with_index[blocks, threads_per_block](data,out,ind,k1,rng_states,timeforward,-gap,Random_walk_out)
        purchase=np.imag(Random_walk_out)
        pattern=np.real(Random_walk_out)
        d=pandas.DataFrame({'purchase':purchase,'pattern':pattern})
        d=d[d.purchase>=0]
        d.loc[d.pattern<0,'pattern']=0
        #d['patternb']="{0:b}".format(d.pattern)
        d['patternb']=d.pattern.apply(lambda x:"{0:b}".format(int(x)))
        d['patternb']=d.patternb.apply(lambda x:str(x).zfill(timeforward))
        x=0
        for i in range(len(history)):
            x=x+(history[len(history)-i-1])*(2**i)
        d.loc[:,'history']=x
        d['bhistory']=d.history.apply(lambda x:"{0:b}".format(int(x)))
        d['bhistory']=d.bhistory.apply(lambda x:str(x).zfill(dyasbefore))
        return(d)
        #return (np.quantile(x,[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
    else:
        return(-1)
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
    dayfromstart=np.max(datasource['dayfromstart'])
    total_purchase=np.sum(datasource[:-1]['pramount'])
    ratio=np.round(total_purchase/(dayfromstart+0.01))
    gap=(datetime.strptime(datebreak,"%Y-%m-%d") -  datetime.strptime(np.max(datasource['prtime']),"%Y-%m-%d")).days
    history_length=(datetime.strptime(datebreak,"%Y-%m-%d") -  datetime.strptime(np.min(datasource['prtime']),"%Y-%m-%d")).days+1
    pramount=datasource.tail(1)['pramount'].item()
    
    if history_length<dyasbefore:
        dyasbefore=history_length  
        
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
    d=estimate_purchase(total_purchase,pramount,gap,dayfromstart,daysafter,transition_matrix,current_state,history,ratio,dyasbefore)
    return(d)


@numba.jit(parallel=True,nopython=True)
def cVestDiscount (data):
    out = np.empty((data.shape[0],6000), dtype=np.int32)
    for i in range(0,data.shape[0]):
        k1=data[np.where((data[:,0]>=float(0.9*data[i,4])) & (data[:,0]<=float(1.1*data[i,4])) & (data[:,3]>=float(0.9*data[i,9])) & (data[:,3]<=float(1.1*data[i,9])) & (data[:,5]>=float(0.9*data[i,10])) & (data[:,5]<=float(1.1*data[i,10])))]
        for j in range(k1[:,6].shape[0]):
            out[i,j]=k1[j,6]
    return out



@numba.jit()
def cVestDiscount__ratio_nonparallel(data):
    out = []
    for i in range(0,data.shape[0]):
        k1=data[np.where((data[:,0]>=float(0.9*data[i,4])) & (data[:,0]<=float(1.1*data[i,4])) & (data[:,3]>=float(0.9*data[i,9])) &\
                         (data[:,3]<=float(1.1*data[i,9])) & (data[:,11]>=float(0.9*data[i,12])) & (data[:,11]<=float(1.1*data[i,12])))]
        out.append(k1[:,6])
    return out


#import multiprocessing
#
#chunks = [(sub_arr)
#              for sub_arr in np.array_split(data, multiprocessing.cpu_count())]
#
#pool = multiprocessing.Pool()
#individual_results = pool.map(cVestDiscount__ratio_nonparallel, chunks)
## Freeing the workers:
#pool.close()
#pool.join()
#
#
#length = max(map(len, individual_results[0]))
#y0=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in individual_results[0]])    
#
#length = max(map(len, individual_results[1]))
#y1=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in individual_results[1]])  
#
#length = max(map(len, individual_results[2]))
#y2=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in individual_results[2]])  
#
#length = max(map(len, individual_results[3]))
#y3=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in individual_results[3]])  
#
#length = max(map(len, individual_results[4]))
#y4=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in individual_results[4]])  
#
#length = max(map(len, individual_results[5]))
#y5=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in individual_results[5]])  
#
#
#
#
#length = 2367
#y0=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in y0])    
#
#length = 2367
#y1=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in y1])  
#
#length = 2367
#y2=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in y2])  
#
#length = 2367
#y3=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in y3])  
#
#length = 2367
#y4=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in y4])  
#
#length = 2367
#y5=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in y5])  
#
#del individual_results
#del chunks
#del x
#del df
#
#out=np.vstack((y0,y1))
#del y0
#del y1
#
#out=np.vstack((out,y2))
#del y2
#
#
#
#y3,y4,y5
#
#
#del y2
#del y3
#del y4
#del y5
#
##@guvectorize(['void(float64[:], float64[:,:],float64[:])'], '(m),(n,m)->()', target='cuda')
##def my_func(data1, data,out):
##    x=0
##    for i in range(0,data.shape[0]):
##        if((data[i,0]>=float(0.85*data1[4])) &\
##           (data[i,0]<=float(1.15*data1[4])) &\
##           (data[i,3]>=float(0.85*data1[9])) &\
##           (data[i,3]<=float(1.15*data1[9])) &\
##           (data[i,11]>=float(0.9*data1[12])) &\
##           (data[i,11]<=float(1.1*data1[12]))):
##            x=x+1
##    out[0]=x
##
##
##data1=data
#out=np.zeros(data.shape[0],np.float64)
#x=my_func(data1,data)

  
out=cVestDiscount__ratio_nonparallel(data)
length = max(map(len, out))
out=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in out])    


ind=np.zeros(out.shape[0])
ind=ind-1
counter=0
for i in range(0,out.shape[0]):
    if len(np.nonzero(out[i])[0])>0:
        ind[i]=counter
        counter=counter+1
out=out[~np.all(out == 0,axis=1)]

out=np.ascontiguousarray(out)
k1=np.ascontiguousarray(k1)
ind=np.ascontiguousarray(ind)
data=np.ascontiguousarray(data)


sample = pandas.read_csv('sim.csv')
quantile=[]
for j in range(sample.shape[0]): #i in range(sample.shape[0])
    userid=sample.iloc[j]['userid']
    GamingServerID=sample.iloc[j]['gamingserverid']
    datebreak='2018-03-12'
    dyasbefore=20
    timeforward=30
    test=grab_userid_data(userid,GamingServerID,datebreak,dyasbefore,timeforward)
    if(type(test)!=int):
        if((len(test['bhistory'][0])>=20)):
            p_value=[]
            for i in range(test.shape[0]):
                n1=IntVector(test.iloc[i]['patternb'])
                n2=IntVector(test.iloc[i]['bhistory'])
                p_value.append(r_f(n1,n2)[0])
            test['pvalue']=p_value
            homogen=test.loc[(test.pvalue > 0.05) , ['purchase']]
            non_homogen=test.loc[(test.pvalue <0.05), ['purchase']]
            
            final_purchase_value=[]
            for i in range(1000):
                if random.uniform(0, 1)<0.1435425:
                    final_purchase_value.append(non_homogen.sample(1)['purchase'].iloc[0])
                else:
                    final_purchase_value.append(homogen.sample(1)['purchase'].iloc[0])
            
            #quantile.append(np.quantile(final_purchase_value,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]))
            quantile.append(final_purchase_value)
        else:
            quantile.append([-1])
    else:
        quantile.append([-1])
    print(j)
    
    
length = max(map(len, quantile))
y=np.array([np.hstack((xi,[0]*(length-len(xi)))) for xi in quantile])


#these are junks not part of the main program

userid_pramount_time_pramount_2_bonus_march_2018_han_2019
userid pramount paamount sice june 2018

df2 = pandas.read_csv('userid_pramount_time_pramount_2_bonus_march_2018_han_2019.csv')
del df2['UserID']
del df2['GamingServerID']
del df2['bonus']
df2=df2.loc[(df2.totalpurchasebefore>= 50)]
df2=df2.sort_values(['pramount','totalpurchasebefore','dayfromstart'], ascending=[True, True,True])
x=range(0,df2.shape[0])
df2['indexx']=x
x=np.zeros(df2.shape[0])
df2['maxx']=x
df2['minn']=x
df2['totalpluspr']=df2['totalpurchasebefore']+df2['pramount']
df2['dayfromstartplus']=df2['dayfromstart']+df2['time']
df2['ratio']=df2['totalpurchasebefore']/df2['dayfromstart']
df2['ratioplus']=df2['totalpluspr']/df2['dayfromstartplus']

data2=df2.values



k1=data2[np.where((data2[:,0]>=(0.9*int(pramount))) & (data2[:,0]<=(1.1*int(pramount))) & (data2[:,3]>=(0.9*total_purchase)) &\
                     (data2[:,3]<=(1.1*total_purchase)) & (data2[:,1]>gap) & (data2[:,1]<=(gap+timeforward)) & (data2[:,11]>=(0.9*ratio)) &\
                     (data2[:,11]<=(1.1*ratio)))]
k2=data2[np.where((data2[:,0]>=(0.9*int(pramount))) & (data2[:,0]<=(1.1*int(pramount))) & (data2[:,3]>=(0.9*total_purchase)) &\
                     (data2[:,3]<=(1.1*total_purchase)) & (data2[:,1]==0) & (data2[:,11]>=(0.9*ratio)) &\
                     (data2[:,11]<=(1.1*ratio)))]
k1=np.vstack((k1,k2))

from random import randint



userid=29183136
GamingServerID=45
datebreak='2018-03-12'
dyasbefore=20
timeforward=30

purchase=[]
pattern=[]
for j in range(20000):
    pramount=42
    total_purchase=1678
    t=0
    ratio=84.0
    k1=data2[np.where((data2[:,0]>=(0.9*int(pramount))) & (data2[:,0]<=(1.1*int(pramount))) & (data2[:,3]>=(0.9*total_purchase)) &\
                     (data2[:,3]<=(1.1*total_purchase)) & (data2[:,11]>=(0.9*ratio)) &\
                     (data2[:,11]<=(1.1*ratio)))]
    total=0
    time=0
    future_states=np.zeros(30)
    index=-1
    bonus=0
    for i in range(0,10000):
        if(k1.shape[0]>0):
            b=randint(0, k1.shape[0]-1)
            index=int(index+k1[b,1])
            if index >= 0:
                if (time+k1[b][1])<=timeforward:
                    future_states[index]=1
                    total=total+k1[b][4]
                    pramount=k1[b][4]
                    time=time+k1[b][1]
                    total_purchase=k1[b][3]+k1[b][0]
                    ratio=k1[b][12]
                    k1=data2[np.where((data2[:,0]>=(0.9*int(pramount))) & (data2[:,0]<=(1.1*int(pramount))) & (data2[:,3]>=(0.9*total_purchase)) &\
                                     (data2[:,3]<=(1.1*total_purchase)) & (data2[:,11]<=(1.1*ratio)) & (data2[:,11]>=(0.9*ratio)))]
                else:
                    break
            else:
                break
        else:
            total=-1
            break
    purchase.append(total)
    c=0
    for i in range(30):
        c=c+(future_states[30-i-1])*(2**i)
    pattern.append(c)
    print(j)
   
d=pandas.DataFrame({'purchase':purchase,'pattern':pattern})
d=d[d.purchase>=0]
d['patternb']=d.pattern.apply(lambda x:"{0:b}".format(int(x)))
d['patternb']=d.patternb.apply(lambda x:str(x).zfill(timeforward))
x=0
for i in range(len(history)):
    x=x+(history[len(history)-i-1])*(2**i)
d.loc[:,'history']=x
d['bhistory']=d.history.apply(lambda x:"{0:b}".format(int(x)))
d['bhistory']=d.bhistory.apply(lambda x:str(x).zfill(dyasbefore))


p_value=[]
for i in range(d.shape[0]):
    n1=IntVector(d.iloc[i]['patternb'])
    n2=IntVector(d.iloc[i]['bhistory'])
    p_value.append(r_f(n1,n2)[0])
d['pvalue']=p_value
homogen=d.loc[(d.pvalue > 0.05) , ['purchase']]
non_homogen=d.loc[(d.pvalue <=0.05), ['purchase']]

final_purchase_value=[]
final_bonus_value=[]
for i in range(1000):
    if random.uniform(0, 1)<0.1435425:
        x=randint(0,non_homogen.shape[0]-1)
        final_purchase_value.append(non_homogen.iloc[x]['purchase'])
    else:
        x=randint(0,homogen.shape[0]-1)
        final_purchase_value.append(homogen.iloc[x]['purchase'])

np.quantile(final_purchase_value,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
np.quantile(d['purchase'],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
