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

df = pandas.read_csv('purchase_time_purchase_purchasecount_purchasebefore_nousd_userid_purchaseocunt_dayfromstart_from_jan_2017.csv')
del df['UserID']
del df['GamingServerID']
df=df.loc[(df.totalpurchasebefore>= 50)]
df=df.sort_values(['pramount','totalpurchasebefore','dayfromstart'], ascending=[True, True,True])
x=range(0,df.shape[0])
df['indexx']=x
x=np.zeros(df.shape[0])
df['maxx']=x
df['minn']=x
#df['totalpluspr09']=0.9*(df['totalpurchasebefore']+df['pramount'])
#df['totalpluspr101']=1.1*(df['totalpurchasebefore']+df['pramount'])
df['totalpluspr']=df['totalpurchasebefore']+df['pramount']
#df['pramount09']=0.9*df['pramount2']
#df['pramount101']=1.1*df['pramount2']
#df['dayfromstart09']=0.9*(df['dayfromstart']+df['time'])
#df['dayfromstart101']=1.1*(df['dayfromstart']+df['time'])
df['dayfromstartplus']=df['dayfromstart']+df['time']
#df['totalpluspr09']=df['totalpluspr09'].astype('int64')
#df['totalpluspr101']=df['totalpluspr101'].astype('int64')
#df['totalpurchasebeforeint']=df['totalpurchasebefore'].astype('int64')

data=df.values
w=df[['pramount','totalpurchasebefore','dayfromstart']]
data=w.values
from numba import jit

def neighbour(row):
    #k1 = df.loc[(df.pramount>= float(0.9*row['pramount2'])) & (df.pramount<= float(1.1*row['pramount2'])) & (df.totalpurchasebefore>= float(0.9*row['totalpluspr'])) & (df.totalpurchasebefore<= float(1.1*row['totalpluspr'])) & (df.dayfromstart>= float(0.9*row['dayfromstartplus'])) & (df.dayfromstart<= float(1.1*row['dayfromstartplus'])), ['indexx']]
    k1=data[np.where((data[:,0]>=float(0.9*row['pramount2'])) & (data[:,0]<=float(1.1*row['pramount2'])) & (data[:,3]>=float(0.9*row['totalpluspr'])) & (data[:,3]<=float(1.1*row['totalpluspr'])) &\
                               (data[:,5]>=float(0.9*row['dayfromstartplus'])) & (data[:,5]<=float(1.1*row['dayfromstartplus'])))]
    return k1[:,6].tolist()

row=df.head(1)




for i in range(20000):
    neighbour(df.iloc[i])
    print(i)



df['indexxx']=df.swifter.apply(neighbour,axis=1)

d = d / d.max(axis=0)
data = data / data.max(axis=0)


dff=df[['pramount2','totalpluspr','dayfromstartplus']]
d=dff.values

from scipy import spatial
x=spatial.cKDTree(d)

distance=x.query(data, k=200)[0][:,:]
ind=x.query(data, k=200)[1][:,:]



#df['indexxx']=df.apply(neighbours,axis=1)
#
#@numba.guvectorize(["void(float64[:],float64)"],"(n)->()")
#def neighbours(row,x):
#    #k1 = df.loc[(df.pramount>= float(0.9*row['pramount2'])) & (df.pramount<= float(1.1*row['pramount2'])) & (df.totalpurchasebefore>= float(0.9*row['totalpluspr'])) & (df.totalpurchasebefore<= float(1.1*row['totalpluspr'])) & (df.dayfromstart>= float(0.9*row['dayfromstartplus'])) & (df.dayfromstart<= float(1.1*row['dayfromstartplus'])), ['indexx']]
#    k1=data[np.where((data[:,0]>=float(0.9*row['pramount2'])) & (data[:,0]<=float(1.1*row['pramount2'])) & (data[:,3]>=float(0.9*row['totalpluspr'])) & (data[:,3]<=float(1.1*row['totalpluspr'])) &\
#                               (data[:,5]>=float(0.9*row['dayfromstartplus'])) & (data[:,5]<=float(1.1*row['dayfromstartplus'])))]
#    return k1[:,6].tolist()
#
#
#neighbours(data)
#
#x=k1[:,6].tolist()
#x = list(map(int, x))
#from itertools import groupby, count
#
#def as_range(iterable): # not sure how to do this part elegantly
#    l = list(iterable)
#    if len(l) > 1:
#        return '{0}-{1}'.format(l[0], l[-1])
#    else:
#        return '{0}'.format(l[0])
#
#
#
#
#w=','.join(as_range(g) for _, g in groupby(sorted(x), key=lambda n, c=count(): n-next(c)))
#
#for g in groupby(sorted(x), key=lambda n, c=count(): n-next(c)):
#    print(g)

@cuda.jit
def neighbour_hood(data,k1,rng_states,time,gap,out):
    thread_id = cuda.grid(1)
    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*k1.shape[0])
    y=k1[randomidx,11]
    randomidy=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*len(y))
    out[thread_id]=complex(-1,y[randomidy])   






@cuda.jit
def Random_walk_Monte_carlo_simulation_2(data,k1,rng_states,time,gap,out):
    thread_id = cuda.grid(1)
    randomidx=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*k1.shape[0])
    y=k1[randomidx,11]
    randomidy=np.int32((xoroshiro128p_uniform_float32(rng_states, thread_id))*len(y))
    out[thread_id]=complex(-1,y[randomidy])   



def estimate_purchase(total_purchase,pramount,gap,ratio,timeforward,transition_matrix,current_state):
    k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)) & (data[:,1]<=timeforward) & (data[:,4]>=(0.9*dayfromstart)) & (data[:,4]<=(1.1*dayfromstart)))]
    if(k1.shape[0]<15):
        k1=data[np.where((data[:,0]>=(0.9*int(pramount))) & (data[:,0]<=(1.1*int(pramount))) & (data[:,2]>=(0.9*total_purchase)) & (data[:,2]<=(1.1*total_purchase)))]
    if(k1.shape[0]>15):
        threads_per_block = 1024
        blocks = 2048
        seed=np.random.randint(1,100)
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed)
        Random_walk_out = np.zeros(threads_per_block * blocks, dtype=np.complex64)
        Random_walk_Monte_carlo_simulation_2[blocks, threads_per_block](data,k1,rng_states,timeforward,-gap,Random_walk_out)
        purchase=np.imag(Random_walk_out)
        pattern=np.real(Random_walk_out)
        d=pandas.DataFrame({'purchase':purchase,'pattern':pattern})
        d=d[d.purchase>=0]
        #d['patternb']="{0:b}".format(d.pattern)
        d['patternb']=d.pattern.apply(lambda x:"{0:b}".format(int(x)))
        d['patternb']=d.patternb.apply(lambda x:str(x).zfill(15))
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
timeforward=30
d.to_csv('pattern_sim.csv', sep=',')
#
#
#
#
#
#
#
#
#
#
#
#result = pandas.DataFrame([(d,tup.pramount, tup.time, tup.purchasecount,
#       tup.totalpurchasebefore, tup.pramount2, tup.dayfromstart,
#       tup.totalpluspr09, tup.totalpluspr101, tup.totalpluspr, tup.pramount09,
#       tup.pramount101,tup.dayfromstart09,tup.dayfromstart101,tup.dayfromstartplus,tup.totalpurchasebeforeint) for tup in df.itertuples() for d in range(tup.totalpluspr09,tup.totalpluspr101+1)])
#
#result.columns=['inttotalpluspr09_totalpluspr101', 'pramount', 'time', 'purchasecount',
#       'totalpurchasebefore', 'pramount2', 'dayfromstart','totalpluspr09', 'totalpluspr101', 'totalpluspr', 'pramount09',
#       'pramount101', 'dayfromstart09', 'dayfromstart101', 'dayfromstartplus',
#       'totalpurchasebeforeint']
#
#del result['totalpluspr09']
#del result['totalpluspr101']
#
#
#
##g=df[['pramount','time','totalpurchasebefore','pramount2','dayfromstart','indexx','totalpluspr','dayfromstartplus','key']]
##df2=df[['key','totalpluspr101','totalpluspr09','indexx']]
##g = g.merge(df2,on=['key'],how='left')
##g = g[(g.totalpluspr >= g.totalpluspr09) & (g.totalpluspr <= g.totalpluspr101)]
#
#
#def cond_merge(g,df2):
#    g = g.merge(df2,on=['key'],how='left')
#    g.columns=['pramount', 'time', 'totalpurchasebefore_x', 'pramount2',
#       'dayfromstart', 'totalpluspr', 'dayfromstartplus',
#       'totalpluspr09', 'totalpluspr101', 'key', 'totalpurchasebefore',
#       'indexx']
#    g = g[(g.totalpurchasebefore >= g.totalpluspr09) & (g.totalpurchasebefore <= g.totalpluspr101)]
#    return g
#
#df2=df[['key','totalpurchasebefore','indexx']]
#g=df[['pramount','time','totalpurchasebefore','pramount2','dayfromstart','totalpluspr','dayfromstartplus','totalpluspr09','totalpluspr101','key']].\
#groupby(['pramount','time','totalpurchasebefore','pramount2','dayfromstart','totalpluspr','dayfromstartplus','totalpluspr09','totalpluspr101','key']).apply(cond_merge,df2)
#
##g = g.merge(df2,on=['key'],how='left')
##g.columns=['pramount', 'time', 'totalpurchasebefore_x', 'pramount2',
##       'dayfromstart', 'indexx_x', 'totalpluspr', 'dayfromstartplus',
##       'totalpluspr09', 'totalpluspr101', 'key', 'totalpurchasebefore',
##       'indexx']
##g = g[(g.totalpurchasebefore >= g.totalpluspr09) & (g.totalpurchasebefore <= g.totalpluspr101)]
#
#
#
#g = g.reset_index(drop=True)
#g=g[['pramount', 'time', 'totalpurchasebefore', 'pramount2', 'dayfromstart',
#       'indexx_x', 'totalpluspr', 'dayfromstartplus','indexx_y']]
#
#g.columns=['pramount', 'time', 'totalpurchasebefore', 'pramount2', 'dayfromstart',
#       'indexx_x', 'totalpluspr', 'dayfromstartplus','indexx']
#
#df2=df[['pramount09','pramount101','indexx']]
#g = g.merge(df2,on=['indexx'],how='left')
#g = g[(g.pramount2 >= g.pramount09) & (g.pramount2 <= g.pramount101)]
#g = g.reset_index(drop=True)
#
#g=g[['pramount', 'time', 'totalpurchasebefore', 'pramount2', 'dayfromstart',
#       'indexx_x', 'totalpluspr', 'dayfromstartplus','indexx']]
#
#df2=df[['dayfromstart09','dayfromstart101','indexx']]
#g = g.merge(df2,on=['indexx'],how='left')
#g = g[(g.dayfromstartplus >= g.dayfromstart09) & (g.dayfromstartplus <= g.dayfromstart101)]
#g = g.reset_index(drop=True)
#g=g[['pramount', 'time', 'totalpurchasebefore', 'pramount2', 'dayfromstart',
#       'indexx_x', 'totalpluspr', 'dayfromstartplus','indexx']]
#
#
#
#
#
#
#
