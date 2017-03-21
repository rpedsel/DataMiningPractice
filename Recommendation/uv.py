import numpy as np
from sys import argv

"""python uv.py input-matrix n m f k"""

if __name__=="__main__":
    
    mat,n,m,f,k = argv[1],int(argv[2]),int(argv[3]),int(argv[4]),int(argv[5])
    lines = [line.strip() for line in open(mat)]
    lines = map(lambda x:[int(y) for y in x.split(",")],lines)
    M = np.empty((n, m,))*np.nan
    for e in lines:
        M[e[0]-1,e[1]-1] = e[2]
    U = np.ones((n, f,))
    V = np.ones((f, m,))
    
    for it in xrange(k):
        for r in xrange(n):
            for s in xrange(f):
                numerator, denominator = 0, 0     
                for j in xrange(m):
                    if not np.isnan(M[r,j]):
                        sumuv = np.dot(U[r,:],V[:,j])-U[r,s]*V[s,j]
                        numerator += (V[s,j]*(M[r,j]- sumuv))
                        denominator += V[s,j]**2
                U[r,s] = numerator/denominator
        for s in xrange(m):
            for r in xrange(f):
                numerator, denominator = 0, 0
                for i in xrange(n):
                    if not np.isnan(M[i,s]):
                        sumuv = np.dot(U[i,:],V[:,s])-U[i,r]*V[r,s]
                        numerator += (U[i,r]*(M[i,s]-sumuv))
                        denominator += U[i,r]**2
                V[r,s] = numerator/denominator
        
        Mfix = np.matmul(U,V)
        error = 0
        count = 0
        for row in xrange(n):
            for col in xrange(m):
                if not np.isnan(M[row,col]):
                    error += (M[row,col]-Mfix[row,col])**2
                    count += 1
        print "%.4f" %np.sqrt(error/count)
