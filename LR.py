from numpy import *
import os
import re
import sys
from collections import Counter
#import math
from math import log,e


# function for the list of stopwords in given file
def getStopWords(stopwords_file):
    stopWords = []
    file = open(stopwords_file)
    stopWords = file.read().strip().split()
    return stopWords


# function to train for MCAP Logistic Regression & returns the weight vector
def LogisticRegTrain(trainFeature, labelList, lmbda, eeta, iterations):
    featureMTX = mat(trainFeature)
    p, q = shape(featureMTX)
    labelMTX = mat(labelList).transpose()
    weight = zeros((q, 1))
    for i in range(iterations):
        predict_condProb = 1.0 / (1 + exp(-featureMTX * weight))
        error = labelMTX - predict_condProb
        weight = weight + eeta * featureMTX.transpose() * error - eeta * lmbda * weight
    return weight


# Function to apply MCAP Logistic Regression and  returns the accuracy
def LogisticRegTest(weight, testFeature, lenTestSpam, lenTestHam):
    featureMTX = mat(testFeature)
    res = featureMTX * weight
    val = 0
    len_allDict = lenTestSpam + lenTestHam
    for i in range(lenTestSpam):
        if (res[i][0] < 0.0):
            val += 1
    i = 0
    for i in range(lenTestSpam + 1, len_allDict):
        if (res[i][0] > 0.0):
            val += 1
    return (float)(val / len_allDict) * 100

#feature vector
def featureVec(distinctWords, dict):
    feature_arr = []
    for f in dict:
        row = [0] * (len(distinctWords))
        for i in distinctWords:
            if (i in dict[f]):
                row[distinctWords.index(i)] = 1
        row.insert(0, 1)
        feature_arr.append(row)
    return feature_arr


# read data files without removing stopWords
def readWithSW(folder):
    files = os.listdir(folder)
    dict = {}
    voc = []
    for f in files:
        file = open(folder + "/" + f, encoding="ISO-8859-1")
        words = file.read()
        WordsCollection = words.strip().split()
        dict[f] = WordsCollection
        voc.extend(WordsCollection)
    return voc, dict


# read data files by removing stopWords
def readWithoutSW(folder, stopWordsF):
    files = os.listdir(folder)
    dict = {}
    voc = []
    stopWords = getStopWords(stopWordsF)
    for f in files:
        file = open(folder + "/" + f, encoding="ISO-8859-1")
        words = file.read()
        words = re.sub('[^a-zA-Z]', ' ', words)
        WordsCollection = words.strip().split()
        reqWords = []
        for word in WordsCollection:
            if (word not in stopWords):
                reqWords.append(word)
        dict[f] = reqWords
        voc.extend(reqWords)
    return voc, dict


if __name__ == "__main__":
    if (len(sys.argv) != 8):
        print("refer readme file,wrong arguments or less number of arguments has been passed")
        sys.exit()
    # read arguments from terminal
    spamTrain = sys.argv[1]
    hamTrain = sys.argv[2]
    spamTest = sys.argv[3]
    hamTest = sys.argv[4]
    stopWords = sys.argv[5]
    stopWords_inclusion = sys.argv[6]
    lmbda = float(sys.argv[7])

    #hard limit on number of iterations and eeta
    iterations = 100
    eeta = 0.1

    # checking to add/remove stopWords
    if (stopWords_inclusion == "yes"):
        spamVocTrain, spamDictTrain = readWithoutSW(spamTrain, stopWords)
        hamVocTrain, hamDictTrain = readWithoutSW(hamTrain, stopWords)
    else:
        spamVocTrain, spamDictTrain = readWithSW(spamTrain)
        hamVocTrain, hamDictTrain = readWithSW(hamTrain)

    spamVocTest, testSpamWords = readWithSW(spamTest)
    hamVocTest, testHamWords = readWithSW(hamTest)

    distinctWords = list(set(spamVocTrain) | set(hamVocTrain))
    finalTrainDict = spamDictTrain.copy()
    finalTrainDict.update(hamDictTrain)

    finalTestDict = testSpamWords.copy()
    finalTestDict.update(testHamWords)

    labelList = []

    for i in range(len(spamDictTrain)):
        labelList.append(0)

    for j in range(len(hamDictTrain)):
        labelList.append(1)

    trainFeature = featureVec(distinctWords, finalTrainDict)
    testFeature = featureVec(distinctWords, finalTestDict)
    lenTest = len(testSpamWords)
    lenHam = len(testHamWords)

    weight = LogisticRegTrain(trainFeature, labelList, lmbda, eeta, iterations)
    accuracy = LogisticRegTest(weight, testFeature, lenTest, lenHam)
    print("lamda=%.3f ,numebr of iterations=%d,eeta=%.3f" % (lmbda, iterations, eeta))
    print("The Accuracy of Logistic Regression is: %.4f" %(accuracy))
