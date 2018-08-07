# -*- coding: utf-8 -*-
# Analyzing dating match data from the ML in Action book from page : 24

# Three classes are provided (people -> didnot like, liked in small doses, liked in large doses)
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr.close()

    #print numberOfLines
    #print returnMat
    #fr = open(filename)
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        #print listFromLine
        returnMat[index, :] = listFromLine[0:3]
        if(listFromLine[-1]=='didntLike'):
            classLabelVector.append((1))
        elif (listFromLine[-1]=='smallDoses'):
            classLabelVector.append((2))
        else:
            classLabelVector.append((3))
        index= index+1

    #print returnMat
    return returnMat, classLabelVector

def main():
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    #print datingDataMat
    #print datingLabels
    #print datingDataMat[:,1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
    #y label
    plt.ylabel('Y Label')
    # x label
    plt.xlabel('X Label')
    plt.show()

main()
