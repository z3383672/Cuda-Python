import numpy as np
import os
import pandas
from datetime import datetime
from scipy import special
os.chdir(r'C:\Users\mhosseini\Desktop\Python app')
from datetime import timedelta
import networkx as nx
df = pandas.read_csv('purchase_time_purchase_purchasecount_purchasebefore.csv')




G = nx.DiGraph()



#node_list=[]
#for i in range(1,df.shape[0]):
#    node_list.append((df.iloc[i].totalpurchasebefore,df.iloc[i].pramount,df.index.values[i]))
#    
#node_list=[]
#for i in range(1,df.shape[0]):
#    node_list.append((df.iloc[i].totalpurchasebefore,df.iloc[i].pramount))    
    
import time
start_time = time.time()

for i in range(1,df.shape[0]):
    if G.has_edge((df.iloc[i].totalpurchasebefore,df.iloc[i].pramount), (df.iloc[i].totalpurchasebefore+df.iloc[i].pramount,df.iloc[i].pramount2)):
        G[(df.iloc[i].totalpurchasebefore,df.iloc[i].pramount)][(df.iloc[i].totalpurchasebefore+df.iloc[i].pramount,df.iloc[i].pramount2)]['weight'] += 1
    else:
        G.add_edge((df.iloc[i].totalpurchasebefore,df.iloc[i].pramount), (df.iloc[i].totalpurchasebefore+df.iloc[i].pramount,df.iloc[i].pramount2), weight=1)


print("--- %s seconds ---" % (time.time() - start_time)) 

leaves = [node for node in G.nodes() if node[1] == 0.0]
for node in leaves:
    G.add_edge(node,(-1000,-1000),weight=1)

start_time = time.time()
paths =  [x for x in nx.all_simple_paths(G, (0.0, 80.0), (-1000,-1000))]
print("--- %s seconds ---" % (time.time() - start_time)) 
weights=[]
for path in paths:
    x=1
    for first, second in zip(path, path[1:]):
        x=x*(G.get_edge_data(first,second)['weight'])
    weights.append(x)

w=[x[-2][0] for x in paths]

#G.add_nodes_from(node_list)
#nx.draw_networkx(G)
#
#w=[path for path in nx.all_simple_paths(G, source, dest)),key=lambda path: get_weight(path)]
#
#G.edges
#
#nx.algorithms.cycles.find_cycle(G,(7373.0, 60.0))
#
#
#for path in nx.all_simple_paths(G, (7373.0, 60.0), (-1000,-1000), cutoff=20):
#    print(path)

w=G.edges((0.0, 80.0))


w=[x[1][1] for x in G.edges((0.0, 80.0))]

nx.write_gpickle(G,'purchase.gpickle')
G=nx.read_gpickle('purchase.gpickle')

x=nx.all_simple_paths(G, (0.0, 80.0), (-1000,-1000))

weights=[]
summ=[]
for i in range(1,10000000):
    w=next(x)
    summ.append(w[-2][0])
    y=0
    for first, second in zip(w, w[1:]):
        y=y+np.log(G.get_edge_data(first,second)['weight'])
    weights.append(y)
    
    
    
    
    
    
    
w=[x[0] for x in leaves]
