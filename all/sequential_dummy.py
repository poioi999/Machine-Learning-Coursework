import numpy as np

seqs = []
for line in open("sequential_train.txt"):
    seqs.append(line.rstrip().split())

# this is a dummy file so we ignore seqs and
# don't attempt to estimate probabilities,
# just spit out arbitrarily chosen values in the right format

# arbitrary distribution
initial_state_prob = np.array([0.2,0.5,0.3])

# values taken from slide L13: 26/44
transition_matrix = np.array([[0.5,0.1,0.4],[0.3,0.1,0.6],[0.01,0.19,0.8]])

# more made up numbers ...
emission_probs = np.array([[0.25,0.25,0.25,0.25],[0.1,0.2,0.3,0.4],[0.5,0.1,0.1,0.3]])

print("Initial state distribution:")
print(initial_state_prob)

print("Transition matrix:")
print(transition_matrix)

print("Emission probabilities:")
print(emission_probs)
