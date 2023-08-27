import numpy as np
import os
import pandas
from datetime import datetime
from scipy import special
os.chdir(r'C:\Users\mhosseini\Desktop\Python app')
from datetime import timedelta
import networkx as nx
df = pandas.read_csv('purchase_time_purchase_purchasecount_purchasebefore.csv')




k1=df.loc[((df.pramount==current_purchase) & (df.totalpurchasebefore>=(0.99*purchase_before)) & (df.totalpurchasebefore<=(1.01*purchase_before)))]
import time
start_time = time.time()
x=[]
times=[]
for i in range(1000):
    purchase_before=112020
    current_purchase=20
    time=0
    for j in range(1000):
        k1 = df.loc[(df.pramount == current_purchase) & (df.totalpurchasebefore==purchase_before), ['pramount2','time']]
        if k1.shape[0]>0:
            k1=k1.sample(1)
            purchase_before=purchase_before+current_purchase
            if ((k1.iloc[0]['pramount2']>0)  & (time+k1.iloc[0]['time']< 13)):
                current_purchase=k1.iloc[0]['pramount2']
                time=time+k1.iloc[0]['time']
            else:
                break
        else:
            break
    print(time)
    x.append(purchase_before)
    times.append(time)
print("--- %s seconds ---" % (time.time() - start_time))     

purchase_before=0.0
current_purchase=80.0


from random import randrange
@vectorize
def selected(elmt):
    retrun elmt in wanted_list

if x[0].shape>0:
               purchase_before=purchase_before+current_purchase
               tmp=2
               current_purchase=2
               if (tmp==0):
                break
           else:
               break

@cuda.jit
def Monet_carlo_simulation(pramount,c):
    tx=cuda.grid(1)
    current_purchase=80.0
    purchase_before=0.0
    x=pramount.astype(np.float64)
    if(tx < c.size):
       c[tx]=(numpy.sum(numpy.where(x == 0)[0])).astype(np.float64)

        
        
data = df.values
data=np.delete(data,0,1)
threadsperblock = 256
blockspergrid = 128
C_global_mem = cuda.device_array(1000)
pramount_mem = cuda.to_device( pramount )

pramount=data[:,0]
putchasetotal=data[:,1]
pramout2=data[:,3]

Monet_carlo_simulation[64, 64](pramount_mem,C_global_mem)
C = C_global_mem.copy_to_host()
print(C)




