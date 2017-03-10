from itertools import combinations
from collections import defaultdict
from pyspark import SparkContext
from sys import argv

def minh(listx,i):
    return min(map(lambda x: (3*x+13*i)%100,listx))

def jaccard(a, b):
    return float(len(set(a) & set(b)))/len(set(a) | set(b))

if __name__ == "__main__": 
    sc = SparkContext(appName="inf553")
    infile = argv[1]
    outfile = argv[2]
    user_list = sc.textFile(infile).map(lambda x: x.split(',')).map(lambda x:
            (x[0],x[1:])).map(lambda x:(int(x[0][1:]),[int(y) for y in x[1]]))

    band1 = user_list.map(lambda x:
            ((minh(x[1],0),minh(x[1],1),minh(x[1],2),minh(x[1],3)),x[0])).groupByKey().map(lambda 
                        x: list(x[1])).filter(lambda x:len(x)>1).flatMap(lambda x:combinations(x,2))
    band2 = user_list.map(lambda x:
            ((minh(x[1],4),minh(x[1],5),minh(x[1],6),minh(x[1],7)),x[0])).groupByKey().map(lambda
                        x: list(x[1])).filter(lambda x:len(x)>1).flatMap(lambda x:combinations(x,2))
    band3 = user_list.map(lambda x:
            ((minh(x[1],8),minh(x[1],9),minh(x[1],10),minh(x[1],11)),x[0])).groupByKey().map(lambda
                        x: list(x[1])).filter(lambda x:len(x)>1).flatMap(lambda x:combinations(x,2))
    band4 = user_list.map(lambda x:
            ((minh(x[1],12),minh(x[1],13),minh(x[1],14),minh(x[1],15)),x[0])).groupByKey().map(lambda
                        x: list(x[1])).filter(lambda x:len(x)>1).flatMap(lambda x:combinations(x,2))
    band5 = user_list.map(lambda x:
            ((minh(x[1],16),minh(x[1],17),minh(x[1],18),minh(x[1],19)),x[0])).groupByKey().map(lambda
                        x: list(x[1])).filter(lambda x:len(x)>1).flatMap(lambda x:combinations(x,2))
    
    C = defaultdict(list)
    C.update(user_list.collect())
            
    band = band1.union(band2).union(band3).union(band4).union(band5).distinct().map(lambda
            x:(x,jaccard(C[x[0]],C[x[1]])))

    part1 = band.map(lambda x: (x[0][0],(x[0][1],x[1])))
    part2 = band.map(lambda x: (x[0][1],(x[0][0],x[1])))

    band = part1.union(part2).groupByKey().map(lambda x: 
            (x[0],sorted(list(x[1]),key=lambda y:y[0]))).map(lambda
                    x:(x[0],sorted(x[1],key=lambda y:y[1],reverse=True))).map(lambda x:
                            (x[0],(x[1])[:5])).map(lambda x:(x[0],sorted([y[0] for y in
                                x[1]]))).sortByKey()
    f = open(outfile,'w')
    for line in band.collect():
        user = "U"+str(line[0])+":"
        sims = map(lambda x: "U"+str(x),line[1])
        f.write(user)
        for i in sims[:-1]:
            f.write(i+",")
        f.write(sims[-1]+"\n")

    f.close()
