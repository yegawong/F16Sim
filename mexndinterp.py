def intMatrix(n, m):
    return [[0] * (m)] * (n)

def doubleMatrix(n, m):
    return [[0.0] * (m)] * (n)

def getHyperCube(X, V, ndinfo):
    indexMatrix = intMatrix(len(ndinfo), 2)
    for i, indexMax in enumerate(ndinfo):
        xmax = X[i][indexMax-1]
        xmin = X[i][0]
        x = V[i]
        if x<xmin or x>xmax:
            print('Point lies out data grid (in getHyperCube)')
        else:
            for j in range(indexMax):
                if x == X[i][j]:
                    indexMatrix[i][0] = indexMatrix[i][1] = j
                    break
                if x == X[i][j+1]:
                    indexMatrix[i][0] = indexMatrix[i][1] = j+1
                    break
                if x>X[i][j] and x<X[i][j+1]:
                    indexMatrix[i][0] = j
                    indexMatrix[i][1] = j+1
                    break
    return indexMatrix

def getLinIndex(indexVector, ndinfo):
    linIndex = 0
    nDimension = len(ndinfo)
    for i in range(nDimension):
        P = 1
        for j in range(i):
            P *= ndinfo[j]
        linIndex += P*indexVector[i]
    return linIndex

def linearInterpolate(T_val, V, X, ndinfo):
    n=len(ndinfo)
    nVertices = 1<<n
    indexVector = [0] * n
    oldT = [0.0] * nVertices
    for i in range(nVertices):
        oldT[i] = T_val[i]
    dimNum = 0
    while n>0:
        m = n-1
        nVertices = 1<<m
        newT = [0.0] * nVertices
        for i in range(nVertices):
            for j in range(m):
                mask = 1<<j
                indexVector[j] =  (mask & i) >> j
            index1 = 0
            index2 = 0
            for j in range(m):
                index1 = index1 + (1<<(j+1))*indexVector[j]
                index2 = index2 + (1<<j)*indexVector[j]
            f1 = oldT[index1]
            f2 = oldT[index1+1]
            if X[dimNum][0] != X[dimNum][1]:
                lambda_val = (V[dimNum]-X[dimNum][0])/(X[dimNum][1]-X[dimNum][0])
                newT[index2] = lambda_val*f2 + (1-lambda_val)*f1
            else:
                newT[index2] = f1
        oldT = [0.0] * nVertices
        for i in range(nVertices):
            oldT[i] = newT[i]
        n = m
        dimNum += 1
    result = oldT[0]
    return result

def interpn(X:list[list], Y:list, x:list, ndinfo:list):
    nDimension = len(ndinfo)
    indexVector = [0] * nDimension
    xPoint = doubleMatrix(nDimension, 2)
    indexMatrix = getHyperCube(X,x,ndinfo)
    nVertices = 1<<nDimension
    T_val = [0.0] * nVertices
    for i in range(nDimension):
        low  = indexMatrix[i][0]
        high = indexMatrix[i][1]
        xPoint[i][0] = X[i][low]
        xPoint[i][1] = X[i][high]
    for i in range(nVertices):
        for j in range(nDimension):
            mask = 1<<j
            val = (mask & i) >> j
            indexVector[j] = indexMatrix[j][val]
        index = getLinIndex(indexVector,ndinfo)
        T_val[i] = Y[index]
    result = linearInterpolate(T_val,x,xPoint,ndinfo)
    return result
