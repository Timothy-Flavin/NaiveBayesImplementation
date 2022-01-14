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

class NBClassifier:

  __nb_type__ = "bernouli"
  __param_dict__ = None

  def __init__(self, nb_type="bernouli", verbose=False):
    if nb_type.lower() != "bernouli" and nb_type.lower != "multinomial":
      raise ValueError(f"Invalid type: '{nb_type}'")

    if(verbose):
      print(f"Creating classifier of type: {nb_type}")
    self.__nb_type__ = nb_type
    return self

  def fit(self, x_train, y_train, x_test, y_test, verbose=False):
    print(f"Fitting naive bayes with type: {self.__nb_type__}")
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    self.__param_dict__ = {}

    for x,y in zip(x_train,y_train):
      if y not in self.__param_dict__:
        self.__param_dict__[y] = np.zeros((x_train.shape[1]))
        if self.__nb_type__ == "bernouli":
          for i in range(x_train.shape[0]):
            if x[i]>0:
              self.__param_dict__[y][i]+=1
        elif self.__nb_type__ == "multinomial":
          for i in range(x_train.shape[0]):
            self.__param_dict__[y][i]+=x[i]
        
