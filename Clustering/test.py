import sys
import pyspark.mllib.linalg as lg
import pyspark.mllib.feature as ft
import numpy as np
from pyspark.sql import SparkSession

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
tfidf = model.transform(tf)
kPoints = tfidf.repartition(1).takeSample(False, k, 1)
kk = tf.repartition(1).takeSample(False, k, 1)
print model.transform(kk[1])
print doc_tf.collect()
print tfidf.collect()
print kPoints
