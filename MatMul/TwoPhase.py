from pyspark import SparkContext
import sys
sc = SparkContext(appName="inf553")
A = sc.textFile(sys.argv[1],10)
B = sc.textFile(sys.argv[2])

rddA = A.map(lambda x: x.split(","))
rddB = B.map(lambda x: x.split(","))
rddA = rddA.map(lambda x: (int(x[1]),(int(x[0]),int(x[2]))))
rddB = rddB.map(lambda x: (int(x[0]),(int(x[1]),int(x[2]))))
rddB = rddB.groupByKey()
rddB = rddB.map(lambda x:(x[0],list(x[1])))
listB = rddB.collect()
rdd = rddA.map(lambda x:(x[1],[i[1] for i in listB if i[0] == x[0]]))
rdd = rdd.flatMap(lambda x:[((x[0][0],y[0]),x[0][1]*y[1]) for y in x[1][0]])
res = rdd.reduceByKey(lambda x,y:x+y).collect()

f = open(sys.argv[3],'w')
for i in res:
    f.write(str(i[0][0])+","+str(i[0][1])+"\t"+str(i[1])+"\n")

f.close()
