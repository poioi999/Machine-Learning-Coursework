from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

models = []

for seed in range(50):
    hmm_model = hmm.CategoricalHMM(n_components=3, n_iter = 500, random_state=seed)
    hmm_model.fit(X, seqLength) 

    print(seed)
    print(hmm_model.startprob_)
    print(hmm_model.transmat_)
    print(hmm_model.emissionprob_)

    print(hmm_model.monitor_.converged)
    print(hmm_model.score(X, seqLength))

    models.append({
        "seed":seed,
        "log_likelihood":hmm_model.score(X, seqLength),
        "converged":hmm_model.monitor_.converged,
    })

df = pd.DataFrame(models)

fig, ax = plt.subplots(figsize=(11, 4.5))

for converged_val, marker, label in [
    (True, "o", "Converged (True)"),
    (False, "X", "Not Converged (False)"),
]:
  sub = df[df["converged"] == converged_val]
  ax.scatter(
      sub["seed"],
      sub["log_likelihood"],
      marker=marker,
      s=55,
      label=label,
  )

ax.set_title("Log-Likelihood by Seed and Convergence Status")
ax.set_xlabel("Seed")
ax.set_ylabel("Log-Likelihood")
ax.legend()
plt.tight_layout()
plt.show()