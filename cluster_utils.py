import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
import scipy.stats
import sklearn.metrics

hotEncodeColors = { 0: 'black', 1: 'blue', 2: 'gray',  3: 'green',4: 'red', 5: 'white'}
hotEncodeTypes = { 1: 'UNKNOWN_SUB_CLASS', 2: 'PRIVATE', 3: 'COMMERCIAL',  4: 'PICKUP',5: 'TRUCK', 6: 'BUS', 7: 'VAN', 8: 'TRACKTOR'}



def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


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
    new_col_name = column_name + "_freq"
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
