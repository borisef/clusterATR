import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster
from scipy.cluster.hierarchy import fclusterdata
import cluster_utils



LOCATE_SIGMA = 5 # mean error of igun
WEIGHT_LOC = 0.5 # distance "expert" weight
WEIGHT_COLOR = 0.125 # color "expert" weight
WEIGHT_CLASS = 1 - WEIGHT_LOC - WEIGHT_COLOR

COLOR_CREDIT = 0.1 # our belief in possibility of most crazy color combination for same target
TYPES_CREDIT = 0.1 # our belief in possibility of most crazy type combination for same target


all_results_in_csv_name = "all_results.csv" # data table with all results
confmType_csv_name = "confmType.csv" # confusion matrix type classifier
confmColor_csv_name = "confmColor.csv" # confusion matrix color classifier

hotEncodeColors = { 0: 'black', 1: 'blue', 2: 'gray',  3: 'green',4: 'red', 5: 'white'}
hotEncodeTypes = { 1: 'UNKNOWN_SUB_CLASS', 2: 'PRIVATE', 3: 'COMMERCIAL',  4: 'PICKUP',5: 'TRUCK', 6: 'BUS', 7: 'VAN', 8: 'TRACKTOR'}


priorsColor = np.ones(shape=(6,1))/6 # prior believes color
priorsClass = np.ones(shape=(8,1))/8 # prior believes types



def ProbabilityFromDistance(d,sigma,pow=1):
    #distance in meters
    #sigma = standard deviation
    #pow - amplify
    pr = np.exp(-np.power(d/sigma,2))
    pr = np.power(pr,pow)
    return pr

def ConvertConfMatrix2ProbMatrix(M, priors = None, credit = 0):
    # M is is NxN, priors - Nx1 (1/N default)

    N = M.shape[0]
    if(priors is None):
        priors = np.ones(shape=(N,1))/N

    M_probs = np.zeros_like(M)
    M_probs_same_class = np.zeros_like(M)
    for i in range(N):
        for j in range(N):
            for i1 in range(N):
                j1 = i1 # same class
                M_probs_same_class[i,j] += priors[i1]*M[i1,i]* priors[i1]*M[i1,j]
    for i in range(N):
        for j in range(N):
            for i1 in range(N):
                for j1 in range(N):
                    M_probs[i,j] += priors[i1]*M[i1,i]* priors[j1]*M[j1,j]

    M_probs_cond = M_probs_same_class/(M_probs+ 0.00001)
    M_probs_cond = M_probs_cond*(1-credit) + credit


    return M_probs_cond

def ProbabilityFromProbMatrix(PM, c1,c2):
    return PM[c1-1,c2-1]


#load data from csv
df = pd.read_csv(all_results_in_csv_name)

#load confusion matrices
conf1 = np.genfromtxt(confmType_csv_name, delimiter=',')
conf2 = np.genfromtxt(confmColor_csv_name, delimiter=',')

#convert to probabilities
probColors = ConvertConfMatrix2ProbMatrix(conf2,priorsColor, COLOR_CREDIT)
probClasses = ConvertConfMatrix2ProbMatrix(conf1,priorsClass, TYPES_CREDIT)


def similarity(x,y):
    #x, y = Frame,x,y,class,color,ObjID
    if(x[0]==y[0]):#same frame
        dis = 1.0
        print(dis)
        return dis
    if(x[5] == y[5] and x[5]>0):#same object
        dis = 0.0
        print(dis)
        return dis

    d = np.power(np.power(x[1] - y[1],2) + np.power(x[2] - y[2],2),0.5)
    prDist = ProbabilityFromDistance(d,LOCATE_SIGMA)
    prCol = probColors[int(x[4]-1), int(y[4]-1)]
    prClass = probClasses[int(x[3]-1), int(y[3]-1)]


    prTotal = WEIGHT_LOC*prDist + WEIGHT_COLOR*prCol + WEIGHT_CLASS*prClass
    print((1-prTotal))
    return (1.0 - prTotal)



fclust1 = fclusterdata(X = df, t = 0.5 , metric=similarity, criterion='distance', method='complete')
numClust = len(np.unique(fclust1))
print(numClust)

df['label'] = fclust1
#aa = cluster.DBSCAN(eps=0.3, min_samples=1, metric= similarity).fit_predict(df)
#aa2 = cluster.OPTICS(min_samples=1,metric = similarity, eps = 0.3).fit_predict(df)

#draw results
cluster_utils.plot_scatter(df,'color','colors.png')
cluster_utils.plot_scatter(df,'class','types.png')
cluster_utils.plot_scatter(df,'label','labels.png')

print("OK")


