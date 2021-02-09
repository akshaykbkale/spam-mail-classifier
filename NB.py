import os
import re
import sys
#import math
from math import log,e
from collections import Counter

all_distinctWords = []
# Naive Bayes on test sets
def NaiveBayesTest(priSpam, priHam, cpSpam, cpHam, spamTest, hamTest):
    global all_distinctWords
    spamHamList = [spamTest, hamTest]
    # print(spamTest)
    val = 0
    for i in range(len(spamHamList)):
        for j in spamHamList[i]:
            priSpam_log = log(priSpam)
            priHam_log = log(priHam)
            for term in spamHamList[i][j]:
                if term in all_distinctWords:
                    priSpam_log = priSpam_log + log(cpSpam[term])
                    priHam_log = priHam_log + log(cpHam[term])
            if (priSpam_log >= priHam_log and i == 0):
                val += 1
            elif (priSpam_log <= priHam_log and i == 1):
                val += 1
    return (float)(val / (len(spamTest) + len(hamTest))) * 100

def mailTest(priSpam, priHam, cpSpam, cpHam,mail):
    global all_distinctWords
    spamHamList = [mail]

    for i in range(len(spamHamList)):
        for j in spamHamList[i]:
            priSpam_log = log(priSpam)
            priHam_log = log(priHam)
            for term in spamHamList[i][j]:
                if term in all_distinctWords:
                    priSpam_log = priSpam_log + log(cpSpam[term])
                    priHam_log = priHam_log + log(cpHam[term])
            if priHam_log > priSpam_log:
                return "not spam"
    return "spam"

# the list of stopwords in a file
def extractStopWords(stopwordsFile):
    stopWords = []
    file = open(stopwordsFile)
    stopWords = file.read().strip().split()
    return stopWords


# read data files by removing stopWords
def readWithoutSW(folder, stopWordsF):
    files = os.listdir(folder)
    voc = []
    dict = {}
    stopWords = extractStopWords(stopWordsF)
    for f in files:
        file = open(folder + "/" + f, encoding="ISO-8859-1")
        reqWords = []
        words = file.read()
        words = re.sub('[^a-zA-Z]', ' ', words)
        words_all = words.strip().split()
        for word in words_all:
            if (word not in stopWords):
                reqWords.append(word)
        dict[f] = reqWords
        voc.extend(reqWords)
    return voc, dict


# read data files without removing stopWords
def readWithSW(folder):
    files = os.listdir(folder)
    dict = {}
    voc = []
    for f in files:
        file = open(folder + "/" + f, encoding="ISO-8859-1")
        words = file.read()
        all_words = words.strip().split()
        dict[f] = all_words
        voc.extend(all_words)

    return voc, dict

def readTextArea(textInput):
    dict = {}
    voc = []
    all_words = textInput.strip().split()
    dict['newFile'] = all_words
    voc.extend(all_words)


    return voc, dict


# training using naive bayes
def NaiveBayesTrain(totalSpamDocs, totalHamDocs, spamVocTrain, hamVocTrain):
    global all_distinctWords
    priSpam = (float)(totalSpamDocs / (totalSpamDocs + totalHamDocs))
    priHam = (float)(totalHamDocs / (totalSpamDocs + totalHamDocs))

    all_spam_dict = Counter(spamVocTrain)
    all_ham_dict = Counter(hamVocTrain)

    spam_totWords = len(spamVocTrain)
    ham_totWords = len(hamVocTrain)

    all_distinctWords = list(set(all_spam_dict) | set(all_ham_dict))
    num_distinctWords = len(all_distinctWords)

    cpSpam = {}
    cpHam = {}

    for term in all_distinctWords:
        count = 0
        if term in all_spam_dict:
            count = all_spam_dict[term]
        cond_probS = (float)((count + 1) / (spam_totWords + num_distinctWords))
        cpSpam[term] = cond_probS

    for term in all_distinctWords:
        count = 0
        if term in all_ham_dict:
            count = all_ham_dict[term]
        cond_probH = (float)((count + 1) / (ham_totWords + num_distinctWords))
        cpHam[term] = cond_probH

    return priSpam, priHam, cpSpam, cpHam


if __name__ == "__main__":
    if (len(sys.argv) != 8):
        print("refer readme file,wrong arguments or less number of arguments has been passed")
        sys.exit()
    #read arguments from terminal
    spamTrain = sys.argv[1]
    hamTrain = sys.argv[2]
    spamTest = sys.argv[3]
    hamTest = sys.argv[4]
    stopWords = sys.argv[5]
    stopWords_inclusion = sys.argv[6]
    mail = sys.argv[7]

    # checking to add/remove stopWords
    if (stopWords_inclusion == "yes"):
        spamVocTrain, spamDictTrain = readWithoutSW(spamTrain, stopWords)
        hamVocTrain, hamDictTrain = readWithoutSW(hamTrain, stopWords)
    else:
        spamVocTrain, spamDictTrain = readWithSW(spamTrain)
        hamVocTrain, hamDictTrain = readWithSW(hamTrain)

    spamVocTest, spamTest = readWithSW(spamTest)
    hamVocTest, hamTest = readWithSW(hamTest)


    # print(spamVocTest)
    # total spam docs
    totalSpamDocs = len(spamDictTrain)
    totalHamDocs = len(hamDictTrain)

    priSpam, priHam, cpSpam, cpHam = NaiveBayesTrain(totalSpamDocs, totalHamDocs, spamVocTrain, hamVocTrain)
    dummy, mail = readWithSW(mail)
    accuracy = NaiveBayesTest(priSpam, priHam, cpSpam, cpHam, spamTest, hamTest)
    res =  mailTest(priSpam, priHam, cpSpam, cpHam,mail)
    print(res)
    print("The Accuracy for Naive bayes = %.4f"% accuracy)
