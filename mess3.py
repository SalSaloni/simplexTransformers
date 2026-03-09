import numpy as np 

class Mess3:

    def __init__(self, epsilon, alpha = 0.3):

        self.epsilon = epsilon
        self.alpha = alpha
        self.stationary = 3
        self.symb = 3

        self.E = np.full((3,3), epsilon)
        np.fill_diagonal(self.E, 1-2*epsilon)

        self.T = np.full((3,3), alpha)
        np.fill_diagonal(self.T, 1-2*alpha)
        self.stationary = np.array([1/3, 1/3, 1/3])

    def sample_sequence(self, length):

        toks = []

        state = np.random.choice(3, p = self.stationary)

        for _ in range(length):
            symbol = np.random.choice(3, p=self.E[:, state])
            toks.append(symbol)
            state = np.random.choice(3, p = self.T[:, state])
        return toks
    
    def forward(self, sequence):
        beliefs = [self.stationary.copy()]
        bel = self.stationary.copy()

        for tok in sequence:
            likelihood = self.E[tok,:]
            updated = bel*likelihood
            updated /=updated.sum()

            bel = self.T@updated
            beliefs.append(bel.copy())

        return np.array(beliefs)
    
def build_dataset(epsilons, n_sequences, seq_len, seed = 42):
    np.random.seed(seed)

    procs = [Mess3(eps) for eps in epsilons]

    seqs = []
    labs = []
    for i in range(n_sequences):
        comp_id = np.random.randint(len(procs))
        s = procs[comp_id].sample_sequence(seq_len)
        seqs.append(s)
        labs.append(comp_id)

    return np.array(seqs), np.array(labs)

if __name__ == '__main__':
    proc = Mess3(epsilon=0.05)
    seq = proc.sample_sequence(16)
    bels = proc.forward(seq)

    print("Sequence:", seq)
    print("Belief at t=0:", bels[0].round(3))
    print("Belief at t=8:", bels[8].round(3))
