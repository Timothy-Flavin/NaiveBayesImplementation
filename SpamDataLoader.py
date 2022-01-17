from mnist import MNIST

def load_train_test(split=0.75):
  mndata = MNIST('.\MNIST')
  xi, xl = mndata.load_training()
  yi, yl = mndata.load_testing()

  return xi, xl, yi, yl

if __name__ == "__main__":
  load_train_test()