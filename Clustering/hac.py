"""
usage: python hca.py inputfile #_of_final_clusters

ex. python hca.py ./docword.enron_s.txt 5

inputfile format explained at: 
https://archive.ics.uci.edu/ml/datasets/Bag+of+Words

"""

from sys import argv
from scipy.sparse import csc_matrix
import numpy as np
from heapq import *
import itertools

pq = []
entry_finder = {}
REMOVED = '<removed-task'
counter = itertools.count()

def add_task(task, priority=0):
    'Add a new task or update the priority of an existing task'
    if task in entry_finder:
        remove_task(task)
    count = next(counter)
    entry = [priority, count, task]
    entry_finder[task] = entry
    heappush(pq, entry)

def remove_task(task):
    'Mark an existing task as REMOVED.  Raise KeyError if not found.'
    entry = entry_finder.pop(task)
    entry[-1] = REMOVED

def pop_task():
    'Remove and return the lowest priority task. Raise KeyError if empty.'
    while pq:
        priority, count, task = heappop(pq)
        if task is not REMOVED:
            del entry_finder[task]
            return task
    raise KeyError('pop from an empty priority queue')

def distance(c1,c2,matrix):
    v1 = matrix[c1]
    v2 = matrix[c2]
    len1 = np.sqrt(v1.power(2).sum())
    len2 = np.sqrt(v2.power(2).sum())
    if len1 == 0 or len2 == 0:
        return 0
    prod = v1.dot(v2.transpose())
    return 1 - prod[0,0] / (len1*len2)

def centroid(cluster,matrix):
    return matrix[cluster,:].mean(0)

if __name__ == "__main__": 
    lines = [line.rstrip('\n').split(" ") for line in open(argv[1])]
    k = int(argv[2])
    
    N = int(lines.pop(0)[0])
    V = int(lines.pop(0)[0])
    lines.pop(0)
    row = map(lambda x: int(x[0])-1,lines)
    col = map(lambda x: int(x[1])-1,lines)
    data = map(lambda x: int(x[2]),lines)
    
    #tf
    M = csc_matrix((data, (row,col)), shape=(2*N-k,V))
    
    #df, idf
    M_df = csc_matrix(([1]*len(data), (row,col)),shape=(N,V))
    df = M_df.sum(0)
    #idf = csc_matrix(map(lambda x: np.log2((N+1.)/(x+1.)),df)[0])
    idf = csc_matrix(np.log2((N+1.)/(df+1.)))
    
    #tf*idf, normalized
    M = idf.multiply(M)
    #euclid = csc_matrix(1./np.sqrt(M.sum(1)))
    euclid = M.power(2).sum(1).transpose().tolist()
    euclid = csc_matrix(map(lambda x: 1./np.sqrt(float(x)) if float(x)!=0 else 0, euclid[0]))
    #euclid = csc_matrix(map(lambda x: 1./np.sqrt(float(x)) if float(x)!=0 else 0,M.sum(1).tolist()))
    #euclid = np.vectorize(lambda x: 1./np.sqrt(x) if x!=0 else x)
    #euclid = csc_matrix(euclid)
    M =euclid.transpose().multiply(M)
    
    #clusters list to record clusters status
    clusters = [[i] for i in range(N)]
    #print M[0]
    #distance(0,1,M)
    for i in range(N):
        for j in range(i+1,N):
            add_task((i,j),distance(i,j,M))
    
    current_count = N
    
    for i in range(N-k):
        c1,c2 = pop_task()
        for p in pq:
            if p[2] != REMOVED:
                p1,p2 = clusters[p[2][0]],clusters[p[2][1]]
                if (set(p1) | set(p2)) & set(clusters[c1]+clusters[c2]): 
                    remove_task((p[2][0],p[2][1]))
        clusters.append(clusters[c1][:]+clusters[c2][:])
        #print clusters
        clusters[c1] = -1
        clusters[c2] = -1
        #print clusters[current_count]
        M[current_count] = centroid(clusters[current_count],M)
        #print clusters
        #print M[current_count]
        for j in range(current_count):
            if clusters[j] != -1:
                add_task((j,current_count),distance(j,current_count,M))
        current_count += 1
    
    for cluster in clusters:
        if cluster != -1:
            dc = ",".join(str(d+1) for d in sorted(cluster))
            print dc

