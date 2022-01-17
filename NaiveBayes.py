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

from json.encoder import INFINITY
import numpy as np
import math
import string 

class NBClassifier:

  __k__ = 1
  __nb_type__ = "bernouli"
  __sum_dict__ = {}
  __cum_doc_len__ = 0
  __x_len__=0
  __text__=False

  __class_list__ = list() # list of classes
  __class_dict__ = {} # {class: c_sum_dict}
  __class_len__ = {} # {class: number of instances of this class}
  __class_doc_len__ = {} # {class: number of "words" in this class}
  __class_prob__ = {} # {c: pc} of probabilities for each class

  def __init__(self, nb_type="bernouli", k=1,verbose=False):
    self.__k__ = k
    if (nb_type.lower() != "bernouli") and (nb_type.lower() != "multinomial"):
      raise ValueError(f"Invalid type: '{nb_type}'")

    if(verbose):
      print(f"Creating classifier of type: {nb_type}")
    self.__nb_type__ = nb_type.lower()

# k is the laplace smoothing number, usually 1
# text is true or false if it is textual data or not
  def fit(self, x_train, y_train, verbose=False, text=False):
    if verbose:
      print(f"Fitting naive bayes with type: {self.__nb_type__}")
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    self.__x_len__ = x_train.shape[0]
    self.__text__=text

    #x_test = np.array(x_test)
    #y_test = np.array(y_test)
    self.__param_dict__ = {}

    for y in y_train:
      self.__class_dict__[y]={}
    self.__class_list__ = list(self.__class_dict__.keys())
    if verbose:
      print(f"Classes found: {self.__class_list__}")
    for i in self.__class_list__:
      self.__class_doc_len__[i]=0
      self.__class_len__[i]=0
      self.__class_prob__[i]=0

    # We need to get the counts of every kind of "word"
    for x, y, in zip(x_train, y_train):
      self.__class_len__[y] = self.__class_len__.get(y,0)+1
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
      if self.__nb_type__ == "bernouli":
        self.__cum_doc_len__ += 1
        self.__class_doc_len__[y] +=1

      for i in list(sdict.keys()):
        if self.__nb_type__ == "bernouli":
          if(sdict[i]>0):
            self.__sum_dict__[i] = self.__sum_dict__.get(i,0)+1
            self.__class_dict__[y][i] = self.__class_dict__[y].get(i,0)+1
          else:
            self.__sum_dict__[i] = self.__sum_dict__.get(i,0)
            self.__class_dict__[y][i] = self.__class_dict__[y].get(i,0)
        elif self.__nb_type__ == "multinomial":
          self.__sum_dict__[i] = self.__sum_dict__.get(i,0)+sdict[i]
          self.__cum_doc_len__ += sdict[i]

          self.__class_dict__[y][i] = self.__class_dict__[y].get(i,0)+sdict[i]
          self.__class_doc_len__[y] += sdict[i]
    # sanity check of doc length
    if(self.__cum_doc_len__ != x_train.shape[0] and self.__nb_type__ == 'bernouli'):
      print(f"Uh oh, cum_dl{self.__cum_doc_len__}, train_x len: {x_train.shape[0]}")
    return self
# bayes equation: P(Class|x) = P(x|Class)P(Class) / P(x)
# P(x) not needed because it is constant

# P(X|class)
  def _px_class(self, x, y, verbose=False):
    pxc = 0
    if verbose:
      print(f"y: {y}, class_dict[y]: {self.__class_dict__[y]}, class_doc_len: {self.__class_doc_len__[y]}")
    if self.__text__:
      for i,xi in enumerate(x):
        pxc += math.log((1.0*self.__class_dict__[y].get(xi,0) + self.__k__) / (self.__class_doc_len__[y]+self.__k__), 10)
    else:
      for i,xi in enumerate(x):
        num_occurences=0
        if xi>0:
          num_occurences = self.__class_dict__[y][i]
        else:  
          num_occurences = self.__class_doc_len__[y] - self.__class_dict__[y][i]
        
        p_temp = (1.0*num_occurences + self.__k__) / (self.__class_doc_len__[y]+self.__k__)
        if verbose:
          print(f"Num occurences: {num_occurences}, p_temp:{p_temp}")
        if self.__nb_type__ == "bernouli" or xi==0:
          pxc += math.log(p_temp,10)
        elif self.__nb_type__ == "multinomial":
          pxc += math.log(p_temp,10)*xi

    #print(f"_px_class: {math.pow(10,pxc)}, x:{x}")
    return pxc

# P(class)
  def _p_class(self, y):
    return math.log(self.__class_len__[y]/self.__x_len__, 10)

# Predict 2d array like object x [instances,features]
  def predict(self, x, verbose=False):
    predictions = list()
    for xi in x:
      predictions.append(self._predict(xi, verbose))
    return predictions

# Prints returns accuracy given data x with labels y
  def score(self, x, y, verbose=False):
    predictions = self.predict(x, verbose=False)
    tot = len(y)
    correct = 0
    for yi, y_pred in zip(y, predictions):
      if yi == y_pred:
        correct+=1
    
    if verbose:
      print(f"{correct}/{tot} correct, {correct/tot}%")
    return correct/tot
  
# Internal predict function that predicts a single instance x
  def _predict(self, x, verbose=False):
    max_p=-INFINITY
    max_i=-1
    for i, c in enumerate(self.__class_list__):
      if verbose:
        print(f"predicting p({c} | {x})")
        print(f"pxc: {self._px_class(x,c, verbose=False)} + p_class: {self._p_class(c)}")
      self.__class_prob__[c] = self._px_class(x,c, verbose=verbose) + self._p_class(c)
      if verbose:
        print(f"class prob: {math.pow(10,self.__class_prob__[c])}")
      if self.__class_prob__[c] > max_p:
        max_p = self.__class_prob__[c]
        max_i = i
    if verbose:
      print(f"Class probabilities: {self.__class_prob__}")
      print(f"max class: i{max_i}:{self.__class_list__[max_i]}")
      print("----------------------------------------------------------")
    return self.__class_list__[max_i]

# Format's input documents into individual words
  def format_docs(self, x, verbose=False):
    exclist = string.punctuation + string.digits
    table_ = str.maketrans('', '', exclist)
    for i,xi in enumerate(x):
      if verbose:
        print(f"String before\n{xi}")
      # remove punctuations and digits from oldtext
      x[i] = xi.translate(table_)
      x[i] = x[i].lower().split()
      if verbose:
        print(f"String after\n{x[i]}")
    return x