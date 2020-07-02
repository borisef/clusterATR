import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster
from scipy.cluster.hierarchy import fclusterdata
from scipy import stats
import cluster_utils
import pickle

from cluster_utils import hotEncodeColors
from cluster_utils import hotEncodeTypes
from cluster_utils import ConvertConfMatrix2ProbMatrix
from cluster_utils import ProbabilityFromDistance



COLOR_CREDIT = 0.1 # our belief in possibility of most crazy color combination for same target
TYPES_CREDIT = 0.1 # our belief in possibility of most crazy type combination for same target

FCLUSTER_THRESHOLD = 0.8 # threshold on fclusterdata , between [0, 1] , small ==> many clusters , large ==> few clusters

#inputs
all_results_in_csv_name = "data/all_results.csv" # data table with all results
confmType_csv_name = "data/confmType.csv" # confusion matrix type classifier
confmColor_csv_name = "data/confmColor.csv" # confusion matrix color classifier



#parameters
LOCATE_SIGMA = 5 # mean error of igun
WEIGHT_LOC = 0.5 # distance "expert" weight
WEIGHT_COLOR = 0.2 # color "expert" weight
WEIGHT_CLASS = 1 - WEIGHT_LOC - WEIGHT_COLOR
FRAME_COUNT_THRESHOLD = 1  # min frames per target

COLOR_CREDIT = 0.1 # our belief in possibility of most crazy color combination for same target
TYPES_CREDIT = 0.1 # our belief in possibility of most crazy type combination for same target

FCLUSTER_THRESHOLD = 0.7 # threshold on fclusterdata , between [0, 1] , small ==> many clusters , large ==> few clusters

priorsColor = np.ones(shape=(6,1))/6 # prior believes color
priorsClass = np.ones(shape=(8,1))/8 # prior believes types/classes

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
        #print(dis)
        return dis
    if(x[5] == y[5] and x[5]>0):#same object
        dis = 0.0
        #print(dis)
        return dis

    d = np.power(np.power(x[1] - y[1],2) + np.power(x[2] - y[2],2),0.5) # euclidean distance
    prDist = ProbabilityFromDistance(d,LOCATE_SIGMA)
    prCol = probColors[int(x[4]-1), int(y[4]-1)]
    prClass = probClasses[int(x[3]-1), int(y[3]-1)]

    #weighted average
    prTotal = WEIGHT_LOC*prDist + WEIGHT_COLOR*prCol + WEIGHT_CLASS*prClass
    #print((1-prTotal))
    return (1.0 - prTotal)


fclust1 = fclusterdata(X = df, t = FCLUSTER_THRESHOLD , metric=similarity, criterion='distance', method='complete')
numClust = len(np.unique(fclust1))
print("Num clusters" + str(numClust))

df['label'] = fclust1
#aa = cluster.DBSCAN(eps=0.3, min_samples=1, metric= similarity).fit_predict(df) # another clustering to consider


df = cluster_utils.AddText(df)


#major voting per cluster
df_final = (df.groupby('label').agg({
    'x': 'median',
    'y': 'median',
    'Frame': ['min', 'max', 'count'],
    'ObjID': 'count',
    'class': [lambda x: stats.mode(x)[0], lambda x: list(x)],
    'color': [lambda x: stats.mode(x)[0], lambda x: list(x)] }) ).reset_index()

df_final.columns = ["_".join(x) for x in df_final.columns.ravel()]

df_final.rename(columns={'class_<lambda_0>':'class',
                         'color_<lambda_0>':'color',
                         'class_<lambda_1>':'all_class',
                         'color_<lambda_1>':'all_color',
                         'x_median':"x",
                         'y_median':'y',
                         "label_":'label'
                         },
                 inplace=True)


# probability distribution functions (color and type)
df_final= cluster_utils.addFrequencies(df_final,df,'color')
df_final= cluster_utils.addFrequencies(df_final,df,'class')



df_final = cluster_utils.AddTextDff(df_final)

#remove outliers
df_final_no_false = df_final[df_final['Frame_count'] >= FRAME_COUNT_THRESHOLD]


#draw results
cluster_utils.plot_scatter(df,'color','results/colors.png')
cluster_utils.plot_scatter(df,'class','results/types.png')
cluster_utils.plot_scatter(df,'label','results/clusters.png', False)
cluster_utils.plot_scatter(df,'label','results/clusters_with_text.png', True)

#draw  with text etc
cluster_utils.plot_scatter(df_final,'class','results/final_with_text.png', True)
#plot nicer
cluster_utils.plot_scatter_final(df_final_no_false,'results/final_with_text_no_false.png', True)

#save csv's
df_final.to_csv("results/clustered.csv")
df_final_no_false.to_csv("results/clustered_without_false.csv")
df.to_csv("results/original_clustered.csv")

#save pickles
df_final.to_pickle("results/clustered.pckl")
df_final_no_false.to_pickle("results/clustered_without_false.pckl")
df.to_pickle("results/original_clustered.pckl")


print("OK")




