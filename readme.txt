1)For compilation go the folder containing the files in cmd and execute


2)FOR NAIVE BAYES:
  python NB.py <Path for spam training files> <Path for ham training files> <Path for test spam files> <Path for ham test files> <stopWords file path> <yes/no to remove stop-words>

  command : python NB.py train\spam train\ham test\spam test\ham stopWords.txt yes

3)FOR LOGISTIC REGRESSION:
  python LR.py <Path for spam training files> <Path for ham training files> <Path for test spam files> <Path for ham test files> <stopWords file path> <yes/no to remove stop-words> <lambda value>

  default values eeta:0.1 and iterations:100

  command : python LR.py train\spam train\ham test\spam test\ham stopWords.txt yes 0.1
