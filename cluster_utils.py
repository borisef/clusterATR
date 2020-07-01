import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
from scipy import stats
import sklearn.metrics
from scipy.spatial import distance

hotEncodeColors = { 0: 'black', 1: 'blue', 2: 'gray',  3: 'green',4: 'red', 5: 'white'}
hotEncodeTypes = { 1: 'UNKNOWN_SUB_CLASS', 2: 'PRIVATE', 3: 'COMMERCIAL',  4: 'PICKUP',5: 'TRUCK', 6: 'BUS', 7: 'VAN', 8: 'TRACKTOR'}
smallestColor = 0
smallestType = 1

def ConvertConfMatrix2ProbMatrix(M, priors = None, credit = 0):
    # M is is NxN
    # priors - Nx1 (1/N default)
    #return pseudo-distance as (1-prob)

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

def ProbabilityFromDistance(d,sigma,pow=1):
    #distance in meters
    #sigma = standard deviation
    #pow - amplify
    pr = np.exp(-np.power(d/sigma,2))
    pr = np.power(pr,pow)
    return pr

def GetFrequencies(arr,column_name):
    elements_count = collections.Counter(arr)
    shift = 0
    if(column_name == "color"):
        refDict = hotEncodeColors
    else:
        refDict = hotEncodeTypes
        shift = 1

    outFreq = np.zeros(shape = len(refDict), dtype= float)
    for key, value in elements_count.items():
        outFreq[key-shift] = value

    outFreq = outFreq / outFreq.sum()
    return outFreq


def addFrequencies(aggDF, refDF, column_name):
    new_col_name = "freq_" + column_name
    mylist = []
    for i in range(aggDF.shape[0]):
        lbl = aggDF["label"][i]
        temp = refDF[refDF["label"] == lbl][column_name]
        fr = GetFrequencies(np.array(temp), column_name)
        mylist.append(fr)
    aggDF[new_col_name] = mylist
    return aggDF


def AddText(df):
    df["text"] = "(F:" + df["Frame"].astype("str") + "," + df["color"].map(hotEncodeColors).astype("str") + "," + df[
        "class"].map(hotEncodeTypes).astype("str") + ",id:"+ df[
        "ObjID"].astype("str") +  ", L:" + df["label"].astype("str") +")"
    return df

def AddTextDff(dff):
    dff["text"] = "(NumFrms:" + dff["Frame_count"].astype("str") + ", " + dff["color"].map(hotEncodeColors).astype("str") + \
                  ", " + dff["class"].map(hotEncodeTypes).astype("str") +")"
    return dff

def plot_scatter(df, grp_by, save2im, withText = False):

    groups = df.groupby(grp_by)
    if(grp_by == "color"):
        hotEncode = hotEncodeColors
    if (grp_by == "class"):
        hotEncode = hotEncodeTypes
    if (grp_by == "label"):
        hotEncode = np.unique(np.array(df['label']))
        hotEncode = np.append([0], hotEncode)
        ss = hotEncode.astype("str")
        hotEncode = ["cluster " + s for s in ss]

    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        if (grp_by == "color"):
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=hotEncode[name], color = hotEncode[name], markeredgecolor="black")
        else:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=hotEncode[name])
    if("objID" in df.columns):
        groups1=df.groupby("ObjID")
        for name, group1 in groups1:
            if(name > 0):
                ax.plot(group1.x, group1.y, marker='', linestyle='--', ms=1)

    XX = np.array(df["x"])
    YY = np.array(df["y"])

    if(withText):
        TT = np.array(df["text"])
        for i, txt in enumerate(TT):
            ax.annotate(txt, (XX[i], YY[i]))

    ax.legend()
    plt.savefig(save2im)
    plt.title("grp_by," + save2im[:-4] )
    plt.show()

def plot_scatter_final(df, save2im, withText = True):


    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling


    XX = np.array(df["x"])
    YY = np.array(df["y"])
    CC = np.array(df["color"].map(hotEncodeColors))
    SS = np.array(df["Frame_count"])
    SS = ((SS - SS.min() + 1.0)/ (SS.max() - SS.min() + 1.0)*50)
    if(withText):
        TT = np.array(df["text"])
        for i, txt in enumerate(TT):
            ax.annotate(txt, (XX[i], YY[i]))
            ax.scatter(XX[i], YY[i], marker='o', c=CC[i], alpha=0.8, edgecolors="black", s = int(SS[i]))

    #ax.legend()
    plt.savefig(save2im)
    plt.title("grp_by," + save2im[:-4] )
    plt.show()


def AggregateTargets(df_targets):
    agg_df_targets = (df_targets.groupby('target').agg({
    'score': ['sum'],
    'subclass': [ lambda x: list(x)],
    'color': [lambda x: list(x)] }) ).reset_index()
    agg_df_targets.columns = ["_".join(x) for x in agg_df_targets.columns.ravel()]
    agg_df_targets.columns = ["target", "total_score", "all_classes", "all_colors"]
    numTargets = agg_df_targets.shape[0]
    numCols = len(hotEncodeColors)
    numTypes = len(hotEncodeTypes)
    listFreqColor = []
    listFreqType = []

    for target in agg_df_targets['target']:
        print(target)
        freqColor = np.zeros(numCols, dtype = float)
        freqType = np.zeros(numTypes, dtype=float)
        df = df_targets[df_targets["target"] == target]
        clrs = np.array(df.color, dtype=int)
        typs = np.array(df.subclass, dtype=int)
        sc = np.array(df.score)
        for i,co in enumerate(clrs):
            freqColor[co - smallestColor] += sc[i]
            freqType[typs[i] - smallestType] += sc[i]
        listFreqColor.append(freqColor)
        listFreqType.append(freqType)

    agg_df_targets["freq_color"] = listFreqColor
    agg_df_targets["freq_class"] = listFreqType

    return(agg_df_targets)



def ComputeDistancesTargets2Clusters(df_T, df_C, wt, bonus_type, bonus_color):
    NC = df_C.shape[0] # num clusters
    NT = df_T.shape[0] # num targets
    D = np.zeros((NC,NT), dtype=float)
    DTypes = np.zeros_like(D, dtype=float)
    DCcolors = np.zeros_like(D, dtype=float)
    for tar in range(NT):
        for clus in range(NC):
            DCcolors[clus,tar]= distance.cosine(df_T["freq_color"][tar] + bonus_color, df_C["freq_color"][clus] + bonus_color)
            DTypes[clus,tar] = distance.cosine(df_T["freq_class"][tar] + bonus_type, df_C["freq_class"][clus] + bonus_type)

    D = DTypes * wt + DCcolors * (1.0 - wt)
    return D

