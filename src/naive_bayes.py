# -*- coding: utf-8 -*-
"""
Multinomial Naive Bayes spam/ham email classifier.

Trains on two folders of plain-text emails (ham and spam), then classifies the
emails in two test folders and reports how many were labelled correctly, both
with and without stop-word removal. See ALGORITHM.md for a walkthrough.

Created on Sat Mar 25 23:40:17 2017
@author: Mohanakrishna
"""

from __future__ import division
import os
import re
import sys
from math import log

# Stop-word list shipped with the repo (<repo>/data/stopwords.txt).
DEFAULT_STOPWORDS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "stopwords.txt")


def folderText(classifier, path):
  """Read every file in `path`; return (number of files, all text joined)."""
  section = " "
  directory = os.listdir(path)
  count = 0
  for everyfile in directory:
    count += 1
    filesPath = path + '/' + everyfile
    with open(filesPath, 'r', encoding='iso-8859-1') as f:
      section += f.read()
  return count, section


def fnPrior(path, classifier):
  """Return (document count, concatenated text) for one class folder.

  The document count is what later becomes the class prior P(class).
  """
  spamham = folderText(classifier, path)
  count = spamham[0]
  text = spamham[1]
  return count, text


def spamConditionalProbablity(hamSection, stopWordsPath):
    """Word likelihoods P(word | class) for the text of ONE class.

    NOTE: despite the name, this is the generic per-class routine; `Train`
    calls it for both ham and spam text. (Name kept from the original code.)

    Uses add-one (Laplace) smoothing:
        P(word | class) = (count(word) + 1) / totalHamTokenCount
    where totalHamTokenCount = sum over vocabulary of (count + 1).

    Returns (counts, probability list, word -> probability, probability
    assigned to words never seen in training).
    """
    global totalHamTokenCount
    totalHamTokenCount = 0
    noshowHamProb = 1
    hamWordsList = Test(hamSection, stopWordsPath)
    hamDictionary = {}
    # Count how often each word occurs.
    for word in hamWordsList:
        if word in hamDictionary:
            hamDictionary[word] += 1.0
        else:
            hamDictionary[word] = 1.0
    # Smoothed denominator: every word's count plus one.
    for key, value in hamDictionary.items():
        totalHamTokenCount += hamDictionary[key] + 1.0
    hamProbList = []
    hamProbDictionary = {}
    for key, value in hamDictionary.items():
        tokenHamProb = (hamDictionary[key] + 1.0) / totalHamTokenCount
        hamProbDictionary[key] = tokenHamProb
        hamProbList.append(tokenHamProb)
    # An unseen word is treated as having count 0 -> (0 + 1) / total.
    noshowHamProb = noshowHamProb / totalHamTokenCount
    return hamDictionary, hamProbList, hamProbDictionary, noshowHamProb


def hamConditionalProbablity(spamSection, stopWordsPath):
    """Same computation as spamConditionalProbablity (kept for compatibility).

    Not called anywhere; `Train` uses spamConditionalProbablity for both classes.
    """
    global totalSpamTokenCount
    totalSpamTokenCount = 0
    noshowSpamProb = 1
    spamWordsList = Test(spamSection, stopWordsPath)
    spamDictionary = {}
    for word in spamWordsList:
        if word in spamDictionary:
            spamDictionary[word] += 1.0
        else:
            spamDictionary[word] = 1.0

    for key, value in spamDictionary.items():
        totalSpamTokenCount += spamDictionary[key] + 1.0
    hamProbDictionary = {}
    spamProbList = []
    for key, value in spamDictionary.items():
        tokenSpamProb = (spamDictionary[key] + 1.0) / totalSpamTokenCount
        hamProbDictionary[key] = tokenSpamProb
        spamProbList.append(tokenSpamProb)
    noshowSpamProb = noshowSpamProb / totalSpamTokenCount
    return spamDictionary, spamProbList, hamProbDictionary, noshowSpamProb


def Test(str, stopWordsPath):
    """Tokenise text: split on spaces, lowercase, drop 1-character tokens.

    If `stopWordsPath` is not None, stop words are removed as well.
    (Quirk preserved from the original: removing from the list while looping
    over it can skip the word right after each removal, so a few stop words
    survive. Changing this would change the published accuracy numbers.)
    """
    rawWordList = re.split(' ', str)
    wordList = [singleToken.lower() for singleToken in rawWordList if len(singleToken) > 1]
    if stopWordsPath is not None:
        with open(stopWordsPath, 'r', encoding='iso-8859-1') as f:
            stopWords = [token.strip() for token in f.readlines()]

        for every_word in wordList:
            if every_word in stopWords:
                wordList.remove(every_word)
    return wordList


def Train(spam, dataPath, stopWordsPath, classifier):
    """Learn the model for one class folder.

    Returns (document count, raw text, word -> P(word|class),
    unseen-word probability, number of tokens).
    """
    prior = fnPrior(dataPath, classifier)
    spamCount = prior[0]
    spamText = prior[1]
    spamWordsList = Test(spamText, stopWordsPath)
    spamList = spamConditionalProbablity(spamText, stopWordsPath)
    return spamCount, spamText, spamList[2], spamList[3], len(spamWordsList)


def TrainMultinomail(ham, hamDataPath, spam, spamDataPath, stopWordsPath):
    """Train on both classes and compute the class priors.

    Returns (ham text, spam text, ham prior, spam prior, ham word
    probabilities, spam word probabilities, ham unseen-word prob,
    spam unseen-word prob).
    """
    spamData = Train(spam, spamDataPath, stopWordsPath, "spam")
    hamData = Train(ham, hamDataPath, stopWordsPath, "ham")
    # Prior = share of training documents belonging to each class.
    totalDocsCount = spamData[0] + hamData[0]
    global spamprior
    global hamprior
    spamprior = spamData[0] / totalDocsCount
    hamprior = hamData[0] / totalDocsCount

    spamVocab = spamData[1]
    hamVocab = hamData[1]

    spamCondProb = spamData[2]
    hamCondprob = hamData[2]
    return hamVocab, spamVocab, hamprior, spamprior, hamCondprob, spamCondProb, hamData[3], spamData[3]


def _scores(tokens, parameters):
    """Return (ham score, spam score) for a token list.

    Each score is log10(prior) + sum(log10(P(token | class))). Logs avoid the
    numeric underflow of multiplying thousands of tiny probabilities.
    """
    hamscore = log(parameters[2], 10)
    spamscore = log(parameters[3], 10)
    hamcondprob = parameters[4]
    spamcondprob = parameters[5]
    hamnoshowprob = parameters[6]
    spamnoshowprob = parameters[7]
    for tokenkey in tokens:
        if tokenkey in hamcondprob:
            hamscore += log(hamcondprob[tokenkey], 10)
        else:
            hamscore += log(hamnoshowprob, 10)
    for tokenkey in tokens:
        if tokenkey in spamcondprob:
            spamscore += log(spamcondprob[tokenkey], 10)
        else:
            spamscore += log(spamnoshowprob, 10)
    return hamscore, spamscore


def hamTest(classifier, dataPath, hamTrainPath, spamTrainPath, stopWordsPath):
  """Classify every email in the ham test folder.

  Returns (correctly classified as ham, wrongly classified as spam).
  Ties go to ham.
  """
  hamsuccesscount = 0
  hamfailurecount = 0
  hamdir = os.listdir(dataPath)
  parameters = TrainMultinomail("ham", hamTrainPath, "spam", spamTrainPath, stopWordsPath)
  for every_file in hamdir:
      with open(dataPath + '/' + every_file, 'r', encoding='iso-8859-1') as f:
          tokens = Test(f.read(), stopWordsPath)
      hamscore, spamscore = _scores(tokens, parameters)
      if hamscore >= spamscore:
          hamsuccesscount += 1
      else:
          hamfailurecount += 1
  return hamsuccesscount, hamfailurecount


def spamTest(classifier, dataPath, hamTrainPath, spamTrainPath, stopWordsPath):
  """Classify every email in the spam test folder.

  Returns (correctly classified as spam, wrongly classified as ham).
  Ties go to spam.
  """
  spamsuccesscount = 0
  spamfailurecount = 0
  spamdir = os.listdir(dataPath)
  parameters = TrainMultinomail("ham", hamTrainPath, "spam", spamTrainPath, stopWordsPath)
  for every_file in spamdir:
      with open(dataPath + '/' + every_file, 'r', encoding='iso-8859-1') as f:
          tokens = Test(f.read(), stopWordsPath)
      hamscore, spamscore = _scores(tokens, parameters)
      if hamscore <= spamscore:
          spamsuccesscount += 1
      else:
          spamfailurecount += 1
  return spamsuccesscount, spamfailurecount


def _report(title, ham, spam):
    """Print the per-class counts and overall accuracy for one run."""
    print(title)
    print("Number of ham files successfully classified:", ham[0])
    print("Number of ham files successfully not classified: ", ham[1])
    print("Number of spam files successfully classified: ", spam[0])
    print("Number of spam files successfully not classified: ", spam[1])
    Accuracy = (ham[0] + spam[0]) / (ham[0] + ham[1] + spam[0] + spam[1]) * 100
    print("The total accuracy is: ", Accuracy)


def mainfn():
    """CLI entry point: run the classifier with, then without, stop-word removal."""
    if len(sys.argv) != 5:
        print("Invalid input arguments. Please specify input as : "
              "python naive_bayes.py trainhampath trainspampath testhampath testspampath")
        sys.exit(1)

    hamTrainPath = sys.argv[1]
    spamTrainPath = sys.argv[2]
    hamTestPath = sys.argv[3]
    spamTestPath = sys.argv[4]

    # Run 1: stop words are removed from the emails.
    stopWordsPath = DEFAULT_STOPWORDS_PATH
    ham = hamTest("", hamTestPath, hamTrainPath, spamTrainPath, stopWordsPath)
    spam = spamTest("", spamTestPath, hamTrainPath, spamTrainPath, stopWordsPath)
    _report("Naive Bayes Classification without stop words:", ham, spam)
    print('\n')

    # Run 2: stop words are kept.
    stopWordsPath = None
    ham = hamTest("", hamTestPath, hamTrainPath, spamTrainPath, stopWordsPath)
    spam = spamTest("", spamTestPath, hamTrainPath, spamTrainPath, stopWordsPath)
    _report("Naive Bayes Classification with stop words:", ham, spam)


if __name__ == "__main__":
    mainfn()
