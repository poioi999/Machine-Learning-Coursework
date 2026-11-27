from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

labelMap = {'A':0, 'B': 1, 'C':2, 'D':3}

seqs = []
for line in open("sequential_train.txt"):
    seqs.append(line.rstrip().split())

X = []
seqLength = []

for seq in seqs:
    string = seq[0]
    label = []
    for x in range(20):
        label.append(labelMap[string[x]])

    X.append(label)
    seqLength.append(len(label))

X = np.array(X).reshape(-1, 1)

bestSeed = 0
bestLogLikelihood = -sys.float_info.max

for seed in range(50):
    hmm_model = hmm.CategoricalHMM(n_components=3, n_iter = 500, random_state=seed)
    hmm_model.fit(X, seqLength) 

    print(seed)

    print("Converged:", hmm_model.monitor_.converged)
    print("Current start probability", hmm_model.startprob_)
    print("Current transition matrix", hmm_model.transmat_)
    print("Current probabilities", hmm_model.emissionprob_)
    print("Current log-likelihood", hmm_model.score(X, seqLength))

    if (hmm_model.score(X, seqLength) > bestLogLikelihood):
        bestLogLikelihood = hmm_model.score(X, seqLength)
        bestModel = hmm_model
        bestSeed = seed

print("Best start probability", bestModel.startprob_)
print("Best transition matrix", bestModel.transmat_)
print("Emission probabilities", bestModel.emissionprob_)
print("Highest log-likelihood", bestModel.score(X, seqLength))
print("Best seed", bestSeed)
