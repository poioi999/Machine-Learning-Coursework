from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt

labelMap = {'A':0, 'B': 1, 'C':2, 'D':3}

seqs = []
for line in open("sequential_train.txt"):
    seqs.append(line.rstrip().split())

X = []
seqLength = []

for seq in seqs:
    label = [labelMap[x] for x in seq]
    X.append(label)
    seqLength.append(len(label))

X = np.array(X).reshape(-1, 1)

hmm_model = hmm.CategorialHMM(n_components=3, n_iter = 50, random_state=42)
hmm_model.fit(X)