from flask import Flask, request, render_template,redirect
from NB import mailTest,readWithSW,readWithoutSW,readTextArea,NaiveBayesTest,NaiveBayesTrain
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/process', methods=['POST'])
def process():
    #
    # if (len(sys.argv) != 8):
    #     print("refer readme file,wrong arguments or less number of arguments has been passed")
    #     sys.exit()
    # read arguments from terminal
    spamTrain = 'train/spam'
    hamTrain = 'train/ham'
    spamTest = 'test/spam'
    hamTest = 'test/ham'
    stopWords = 'stopWords.txt'
    stopWords_inclusion = 'yes'
    mail = request.form['text']

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
    dummy, mail = readTextArea(mail)
    accuracy = NaiveBayesTest(priSpam, priHam, cpSpam, cpHam, spamTest, hamTest)
    res = mailTest(priSpam, priHam, cpSpam, cpHam, mail)
    print(res)
    # print("The Accuracy for Naive bayes = %.4f" % accuracy)

    # return 'You entered: {}'.format(request.form['text'])
    return render_template('index.html',res = res)


