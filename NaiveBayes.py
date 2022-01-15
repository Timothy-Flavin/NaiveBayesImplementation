# Implementation of NaiveBayes classifier
# arguments:
#
# nb_type: String specifying which classifier to use
#   "bernouli": multivariate Bernoulli 
#   "multinomial": multinomial event
# 
# x_train: 2 dimensional array of the form [item][attributes]
# y_train: list of numerical labels with the form [label]
# x_test, x_train: Same format but for testing

import numpy as np
import math

class NBClassifier:

  __nb_type__ = "bernouli"
  __sum_dict__ = {}
  __cum_doc_len__ = 0

  __class_list__ = list() # list of classes
  __class_dict__ = {} # {class: c_sum_dict}
  __class_len__ = {} # {class: total class len}
  __class_prob__ = list() # list of probabilities for each class

  def __init__(self, nb_type="bernouli", verbose=False):
    if nb_type.lower() != "bernouli" and nb_type.lower != "multinomial":
      raise ValueError(f"Invalid type: '{nb_type}'")

    if(verbose):
      print(f"Creating classifier of type: {nb_type}")
    self.__nb_type__ = nb_type.lower()
    return self

  def _fit_bernouli(self, x_train, y_train, k=1, verbose=False):
    print("bernouli fit")
  def _fit_multinomial(self, x_train, y_train, k, verbose):
    print("multinomial fit")

# k is the laplace smoothing number, usually 1
# text is true or false if it is textual data or not
  def fit(self, x_train, y_train, k=1, verbose=False, text=False):
    print(f"Fitting naive bayes with type: {self.__nb_type__}")
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #x_test = np.array(x_test)
    #y_test = np.array(y_test)
    self.__param_dict__ = {}

    for y in y_train:
      self.__class_dict__[y]={}
    self.__class_list__ = list(self.__class_dict__.keys())
    if verbose:
      print(f"classes: {self.__class_list__}")
    for i in self.__class_list__:
      self.__class_len__[i]=0
      self.__class_prob__[i]=0

    # We need to get the counts of every kind of "word"
    for x, y, in zip(x_train, y_train):
      sdict = {}

      # If text add 1 per word of that type found, if not add the value at that pixel
      for i,xi in enumerate(x):
        if text:
          sdict[xi] = sdict.get(xi, 0) + 1  
        else:
          sdict[i] = sdict.get(i, 0) + xi 
      # Accumulate the number of each "word" or pixel to the overall sum dictionary and 
      # class specific sum dictionary and accumulate the total number of docs if bernouli
      # or the number of "words" if multinomial 
      for i in list(sdict.keys()):
        if self.__nb_type__ == "bernouli":
          self.__sum_dict__[i] = self.__sum_dict__.get(i,0)+1
          self.__cum_doc_len__ += 1

          self.__class_dict__[y][i] = self.__class_dict__[y].get(i,0)+1
          self.__class_len__ +=1

        elif self.__nb_type__ == "multinomial":
          self.__sum_dict__[i] = self.__sum_dict__.get(i,0)+sdict[i]
          self.__cum_doc_len__ += sdict[i]

          self.__class_dict__[y][i] = self.__class_dict__[y].get(i,0)+sdict[i]
          self.__class_len__ += sdict[i]
    #sanity check of doc length
    if(self.__cum_doc_len__ != x_train.shape[0] and self.__nb_type__ == 'bernouli'):
      print(f"Uh oh, cum_dl{self.__cum_doc_len__}, train_x len: {x_train.shape[0]}")
    

# bayes equation: P(Class|x) = P(x|Class)P(Class) / P(x)

  def Px(self,x):
    logpx = 0
    for i in x:
      logpx+=1

  def Pspam_x(self,x):
    print("spam given x")