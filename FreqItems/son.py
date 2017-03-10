from itertools import chain
from itertools import ifilter, imap
from collections import defaultdict
from collections import Counter,deque
from itertools import combinations
from pyspark import SparkContext
import sys

def superset(sets, m):
    #print "--------------superset-------------------"
    if m == 2:
        return list(combinations(sets,m))
    sets = combinations(sets,m)
    count = Counter()
    returnlist = []
    for sett in sets:
        count.clear()
        count.update(chain(*sett))
        if len(count) == m and all(v == m-1 for v in count.values()):
            returnlist.append(tuple(sorted(count.keys())))
    #print "--------------superset done-------------------"
    return returnlist


def apriori(chunk,srate):
    #print "----------------apriori-------------------"
    s = srate*len(chunk)
    returnlist = []
    C = defaultdict(int)
    L = []
    stop = False
    n = 1
    # for first pass
    for basket in chunk:
        for item in basket:
            C[item] += 1
    for k,v in C.iteritems():
        if v >= s:
            L.append(k)
    returnlist += L
    #remaining passes
    while not stop:
        n += 1
        C = defaultdict(int)
        #L = superset(L,n)
        for basket in chunk:
            tmp = list(combinations(basket,n))
            for t in tmp:
                itemset = list(combinations(t,n-1))
                if (n == 2 and all(it[0] in L for it in itemset)) or all(it in L for it in itemset): C[t]+=1
            # for itemset in tmp:
            #     if itemset in L:
            #         C[itemset] += 1
        L = []
        for k,v in C.iteritems():
            if v >= s:
                L.append(k)
        returnlist += L
        if len(L) < n+1:
            stop = True
    return returnlist

def subsets(bask):
    tmp = chain(*[combinations(bask,n) for n in range(1,len(bask)+1)])
    return chain(bask,tmp)

def in_basket(candidate,basket):
    return all(map(lambda x: x in basket, candidate))

def basketcount(chunk,candidate):
    cand = map(lambda x: (x,) if isinstance(x,int) else x,candidate)
    count = defaultdict(int)
    for c in cand:
        if len(c) == 1:
            count[c[0]] = sum(map(lambda x: in_basket(c,x),chunk))
        else:
            count[c] = sum(map(lambda x: in_basket(c,x),chunk))
    return list(count.iteritems())

def basketcount_2(chunk, candidate):
    #print "-------------------basketcount----------------------"
    C = defaultdict(int)
#    C = Counter()
    #print "generate...."
    all_set = ifilter(lambda x: x in candidate,chain(*[subsets(basket) for basket in chunk]))
    #print "count"
    for sett in all_set:
        C[sett]+=1
#    C.update(all_set)
    #print "finish counting"
    return list(C.iteritems())

sc = SparkContext(appName="inf553")
srate = sys.argv[2]
allbaskets = sc.textFile(sys.argv[1],3).map(lambda x: x.split(",")).map(lambda x: [int(i) for i in x]).cache()

def run_apriori(it):
    return apriori(list(it),float(srate))

phase1 = allbaskets.mapPartitions(run_apriori).distinct().collect()
#phase1 = allbaskets.mapPartitions(run_apriori).distinct()
#print "--------------candidate----------------"
#print phase1
def run_basketcount(it):
    return basketcount(list(it),phase1)

f = open(sys.argv[3],'w')

s = float(srate)*allbaskets.count()
phase2 = allbaskets.mapPartitions(run_basketcount)
#print "-------------result collect-------------"
phase2 = phase2.reduceByKey(lambda x,y:x+y).filter(lambda x: x[1]>=s).map(lambda x: x[0]).collect()
for itemset in phase2:
    if type(itemset) == int:
        f.write(str(itemset)+"\n")
    else:
        st = ",".join(map(str,itemset))
        f.write(st+"\n")
f.close()
