import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

hotEncodeColors = { 0: 'black', 1: 'blue', 2: 'gray',  3: 'green',4: 'red', 5: 'white'}
hotEncodeTypes = { 1: 'UNKNOWN_SUB_CLASS', 2: 'PRIVATE', 3: 'COMMERCIAL',  4: 'PICKUP',5: 'TRUCK', 6: 'BUS', 7: 'VAN', 8: 'TRACKTOR'}

def AddText(df):
    df["text"] = "(F:" + df["Frame"].astype("str") + ", C:" + df["color"].astype("str") + ", SubCl" + df[
        "class"].astype("str") + ",id:"+ df[
        "ObjID"].astype("str") + ")"
    return df

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
