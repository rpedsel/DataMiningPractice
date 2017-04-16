import sys
import pyspark.mllib.linalg as lg
import pyspark.mllib.feature as ft
import numpy as np
from pyspark.sql import SparkSession

def add(v1, v2):
    """Add two sparse vectors
    >>> v1 = Vectors.sparse(3, {0: 1.0, 2: 1.0})
    >>> v2 = Vectors.sparse(3, {1: 1.0})
    >>> add(v1, v2)
    SparseVector(3, {0: 1.0, 1: 1.0, 2: 1.0})
    """
    assert isinstance(v1, lg.SparseVector) and isinstance(v2, lg.SparseVector)
    assert v1.size == v2.size
    # Compute union of indices
    indices = set(v1.indices).union(set(v2.indices))
    # Not particularly efficient but we are limited by SPARK-10973
    # Create index: value dicts
    v1d = dict(zip(v1.indices, v1.values))
    v2d = dict(zip(v2.indices, v2.values))
    zero = np.float64(0)
    # Create dictionary index: (v1[index] + v2[index])
    values =  {i: v1d.get(i, zero) + v2d.get(i, zero)\
       for i in indices\
       if v1d.get(i, zero) + v2d.get(i, zero) != zero}

    return lg.SparseVector(v1.size, values)

def divide(v,n):
    vd = dict(zip(v.indices,v.values))
    values = {i:vd.get(i)/n for i in v.indices}
    return lg.SparseVector(v.size, values)

def subtract(v1,v2):
    v1d = dict(zip(v1.indices,v1.values))
    v2d = dict(zip(v2.indices,v2.values))
    zero = np.float64(0)
    values = {i: v1d.get(i, zero) - v2d.get(i, zero)\
            for i in indices\
            if v1d.get(i,zero)+v2d.get(i,zero) != zero}
    return lg.SparseVector(v1.size,values)

def closestPoint(p,centers):
    bestIndex = 0
    closest = float("-inf")
    for i in range(len(centers)):
        tempDist = (p.dot(centers[i])/(p.norm(2)*centers[i].norm(2)))
        if tempDist > closest:
            closest = tempDist
            bestIndex = i
    return bestIndex


if __name__ == "__main__":

    spark = SparkSession\
            .builder\
            .appName("PythonKMeans")\
            .getOrCreate()
    
    lines = spark.read.text(sys.argv[1])\
            .rdd.map(lambda r: r[0])\
            .map(lambda x: x.split(" "))\
            .map(lambda x: [int(i) for i in x])
    k = int(sys.argv[2])
    header = lines.take(2)
    N,V = header[0][0],header[1][0]
    
    doc_tf = lines.filter(lambda x: len(x)>1)\
            .map(lambda x: (x[0]-1,(x[1]-1,x[2])))\
            .groupByKey()\
            .map(lambda x: (x[0],lg.SparseVector(V,x[1])))\
            .sortByKey()\
    
    tf = doc_tf.map(lambda x: x[1])
    idf = ft.IDF()
    model = idf.fit(tf)
    tfidf = model.transform(tf).\
            map(lambda x: lg.SparseVector(x.size,x.indices,x.values/x.norm(2)))\
            .cache()
    kPoints = tfidf.repartition(1).takeSample(False, k, 1)
    convergeDist = float(sys.argv[3])
    tempDist = 1.0
    
    while tempDist > convergeDist:
        closest = tfidf.map(lambda p: (closestPoint(p,kPoints), (p,1)))
        pointStats = closest.reduceByKey(lambda p1_c1,p2_c2:
                (add(p1_c1[0],p2_c2[0]),p1_c1[1]+p2_c2[1]))
        newPoints = pointStats.map(lambda st: (st[0],divide(st[1][0],st[1][1]))).collect()
        tempDist = sum(np.sqrt(kPoints[iK].squared_distance(p)) for (iK,p) in newPoints)
        #tempDist = sum(kPoints[iK].squared_distance(p) for (iK,p) in newPoints)
        for (iK,p) in newPoints:
            kPoints[iK] = p
    
    nz = [v.numNonzeros() for v in kPoints]
    
    f = open(sys.argv[4],'w')
    for i in nz:
        f.write(str(i)+"\n")
    f.close()
