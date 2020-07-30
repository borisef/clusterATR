import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster
from scipy.cluster.hierarchy import fclusterdata
from scipy import stats

from pymap3d.vincenty import vdist as lla_dist
import cluster_utils
import pickle

from cluster_utils import hotEncodeColors
from cluster_utils import hotEncodeTypes
from cluster_utils import ConvertConfMatrix2ProbMatrix
from cluster_utils import ProbabilityFromDistance, ReadCSV_or_TXT, euclidean_dist



COLOR_CREDIT = 0.05 # our belief in possibility of most crazy color combination for same target
TYPES_CREDIT = 0.05 # our belief in possibility of most crazy type combination for same target

IS_LLA = True

#inputs
#all_results_in_csv_name = "data/all_results.csv" # data table with all results
#input_data_type = "type1" # like all_results.csv

all_results_in_csv_name = "data/recogs1.txt" # data table with all results
input_data_type = "type2" # like all_results.csv


confmType_csv_name = "data/confmType.csv" # confusion matrix type classifier
confmColor_csv_name = "data/confmColor1.csv" # confusion matrix color classifier




#parameters
LOCATE_SIGMA = 5 # mean error of igun

FCLUSTER_THRESHOLD = 0.7 # threshold on fclusterdata , between [0, 1] , small ==> many clusters , large ==> few clusters

WEIGHT_LOC = 0.7 # distance "expert" weight
WEIGHT_COLOR = 0.15 # color "expert" weight
WEIGHT_CLASS = 1 - WEIGHT_LOC - WEIGHT_COLOR
FRAME_COUNT_THRESHOLD = 2  # min frames per target
HARD_DISTANCE_THRESHOLD = 20 # min distance for "no chance it is the same car" (unless same ID)

COLOR_CREDIT = 0.1 # our belief in possibility of most crazy color combination for same target
TYPES_CREDIT = 0.1 # our belief in possibility of most crazy type combination for same target


priorsColor = np.ones(shape=(9,1))/7 # prior believes color
priorsColor[[0,2]]= 0  #unknown, silver

priorsClass = np.ones(shape=(8,1))/8 # prior believes types/classes

#load data from csv
df = ReadCSV_or_TXT(all_results_in_csv_name, input_data_type)


#load confusion matrices
conf1 = np.genfromtxt(confmType_csv_name, delimiter=',')
conf2 = np.genfromtxt(confmColor_csv_name, delimiter=',')

#convert to probabilities
probClasses = ConvertConfMatrix2ProbMatrix(conf1,priorsClass, TYPES_CREDIT)
probColors = ConvertConfMatrix2ProbMatrix(conf2,priorsColor, COLOR_CREDIT)





def similarityType1(x,y,is_lla=IS_LLA):
    #x, y = Frame,x (or lat) ,y (or lon),class,color,ObjID

    if(x[0]==y[0]):#same frame
        dis = 1.0
        #print(dis)
        return dis
    if(x[5] == y[5] and x[5]>0):#same object
        dis = 0.0
        #print(dis)
        return dis

    d = euclidean_dist(x[1], y[1], x[2], y[2]) if not is_lla else \
            lla_dist(x[1], x[2], y[1], y[2])[0]

    if(d > HARD_DISTANCE_THRESHOLD):
        dis = 1.0
        # print(dis)
        return dis

    prDist = ProbabilityFromDistance(d,LOCATE_SIGMA)
    prCol = probColors[int(x[4])-1, int(y[4])-1]
    prClass = probClasses[int(x[3]-1), int(y[3]-1)]

    #weighted average
    prTotal = WEIGHT_LOC*prDist + WEIGHT_COLOR*prCol + WEIGHT_CLASS*prClass
    #print((1-prTotal))
    return (1.0 - prTotal)

# def similarityType2(x,y):
#     #x, y = frame_idx target_id ts daytime X Y Z lat lon x y w h clr clr_score cls cls_score
#     if(x[0]==y[0]):#same frame
#         dis = 1.0
#         #print(dis)
#         return dis
#     if(x[1] == y[1] and x[1]>0):#same object
#         dis = 0.0
#         #print(dis)
#         return dis
#
#     d = np.power(np.power(x[4] - y[4],2) + np.power(x[5] - y[5],2),0.5) # euclidean distance
#     print(d)
#     prDist = ProbabilityFromDistance(d,LOCATE_SIGMA)
#     prCol = probColors[int(x[13]-1), int(y[13]-1)] # clr
#     prClass = probClasses[int(x[15]-1), int(y[15]-1)] # cls
#
#     #weighted average
#     prTotal = WEIGHT_LOC*prDist + WEIGHT_COLOR*prCol + WEIGHT_CLASS*prClass
#     #print((1-prTotal))
#     return (1.0 - prTotal)



if(input_data_type is "type1"):
    similarity = similarityType1
    df4clust = df
else:
    similarity = similarityType1 # TODO define your own
    df4clust =  df[['frame_idx', "lat", "lon", "cls", "clr", "target_id"]].copy()



fclust1 = fclusterdata(X = df4clust, t = FCLUSTER_THRESHOLD , metric=similarity, criterion='distance', method='complete')
numClust = len(np.unique(fclust1))
print("Num clusters" + str(numClust))

df['label'] = fclust1
df4clust['label'] = fclust1
#aa = cluster.DBSCAN(eps=0.3, min_samples=1, metric= similarityType1).fit_predict(df) # another clustering to consider


df = cluster_utils.AddText(df)

loc1 = 'X'
loc2 = 'Y'
if 'lat' in df.columns:
    loc1 = 'lat'
    loc2 = 'lon'

#major voting per cluster
df_final = (df.groupby('label').agg({
    loc1: ['median', 'min', 'max'],
    loc2: ['median', 'min', 'max'],
    'frame_idx': ['min', 'max', 'count'],
    'target_id': 'count',
    'ts': ['min', 'count'],
    'cls': [lambda x: stats.mode(x)[0], lambda x: list(x)],
    'clr': [lambda x: stats.mode(x)[0], lambda x: list(x)] }) ).reset_index()

df_final.columns = ["_".join(x) for x in df_final.columns.ravel()]

df_final.rename(columns={'cls_<lambda_0>':'cls',
                         'clr_<lambda_0>':'clr',
                         'cls_<lambda_1>':'all_cls',
                         'clr_<lambda_1>':'all_clr',
                         str(loc1+'_median'):loc1,
                         str(loc2+'_median'):loc2,
                         "label_":'label'
                         },
                 inplace=True)


# probability distribution functions (color and type)
df_final= cluster_utils.addFrequencies(df_final,df,'clr')
df_final= cluster_utils.addFrequencies(df_final,df,'cls')



df_final = cluster_utils.AddTextDff(df_final)
df_final = cluster_utils.AddClusterRange(df_final)
#remove outliers
df_final_no_false = df_final[df_final['frame_idx_count'] >= FRAME_COUNT_THRESHOLD]


#draw results
cluster_utils.plot_scatter(df,'clr','results/colors.png')
cluster_utils.plot_scatter(df,'cls','results/types.png')
cluster_utils.plot_scatter(df,'label','results/clusters.png', False)
cluster_utils.plot_scatter(df,'label','results/clusters_with_text.png', True)
#cluster_utils.plot_scatter(df,'target_id','results/target_ids.png', False)


#draw  with text etc
cluster_utils.plot_scatter(df_final,'cls','results/final_with_text.png', True)
#plot nicer
cluster_utils.plot_scatter_final(df_final_no_false,'results/final_with_text_no_false.png', True)
cluster_utils.plot_scatter_final(df_final_no_false,'results/final_with_range.png', False, "range")

#save csv's
df_final.to_csv("results/clustered.csv")
df_final_no_false.to_csv("results/clustered_without_false.csv")
df.to_csv("results/original_clustered.csv")

#save pickles
df_final.to_pickle("results/clustered.pckl")
df_final_no_false.to_pickle("results/clustered_without_false.pckl")
df.to_pickle("results/original_clustered.pckl")


print("OK")




