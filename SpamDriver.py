import NaiveBayes
import SpamDataLoader
import time
import pandas as pd

x,y,xt,yt = SpamDataLoader.load_train_test()

x_samps = [50,100,500,2000,-1]
y_samps = [50,100,500,750,-1]
ks = [1,2,5,15]
nb_types=['bernouli', 'multinomial']
job_results = pd.DataFrame(columns=['nb_type','num_train', 'num_test', 'k', 'train_acc','test_acc', 'fit_time', 'train_s_time','test_s_time'])

b = NaiveBayes.NBClassifier()
x = b.format_docs(x)
xt = b.format_docs(xt)

if True:
  for t in nb_types:
    for k in ks:
      for i,j in zip(x_samps, y_samps): 
        start = time.time()
        bayes = NaiveBayes.NBClassifier(k=k,nb_type=t)
        bayes = bayes.fit(x[0:i], y[0:i], verbose=False, text=True)
        fit = time.time()-start
        #print(f"Final class predicion: {bayes.predict(x[0:100], verbose=False)}")
        acc1 = bayes.score(x[0:i], y[0:i], verbose=True)
        score1 = time.time()-(fit+start)
        acc2 = bayes.score(xt[0:j], yt[0:j], verbose=True)
        score2 = time.time()-(score1+fit+start)
        job_results.loc[len(job_results.index)] = [t,i,j,k,acc1,acc2,fit,score1,score2]
        #print(job_results)
        job_results.to_csv("SPAM_Results_in_progress.csv")
        #print(f"Num Train Samples: {i}, Num Test Samples: {j}, k: {k}")
        #print(f"Train accuracy: {acc1}, Test Accuracy: {acc2}")
        #print(f"Fit time: {fit}, Score Train time: {score1}, Score Test Time: {score2}")
  job_results.to_csv("SPAM_Results_final.csv")
print(job_results)
x,y,xt,yt = SpamDataLoader.load_train_test(1.0)
print(len(x))
b = NaiveBayes.NBClassifier(nb_type="multinomial")
x = b.format_docs(x)
xt = b.format_docs(xt)
scores, mean, std = b.cross_val(x, y, text=True)
print(f"scores: {scores}, mean: {mean}, std: {std}")