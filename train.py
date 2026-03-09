import torch 
import torch.nn as nn
import numpy as np
from mess3 import Mess3, build_dataset
from model import SmallTransformer

#hyperparameters
EPSILONS = [0.05, 0.15, 0.25]
N_SEQUENCES=3000
SEQ_LEN=16
BATCH_SIZE=64
EPOCHS=30
LR=1e-3
D_MODEL=64
N_LAYERS=2
N_HEADS=2
DEVICE="cuda" if torch.cuda.is_available() else "cpu"


def train():
    #convert numpy arr to pytorch tensor
    seqs, labels = build_dataset(EPSILONS, N_SEQUENCES,SEQ_LEN)
    seqs_tensor = torch.tensor(seqs, dtype = torch.long)

    #uses x to predict y. x components accumulates!
    X=seqs_tensor[:,:-1]
    Y=seqs_tensor[:,1:]

    model = SmallTransformer(vocab_size=3, d_model = D_MODEL, n_heads = N_HEADS, n_layers = N_LAYERS, context_len=SEQ_LEN-1, d_ff=128).to(DEVICE)
    #adam optimizer to adapt the learning rate per parameter
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    #takes raw logits and the true token id and computes -log P(correct token). Loss in inverted to p. So if model
    #assigns p of 0.9 then loss is low vs p of 0.1 then loss is high.
    loss_fn = nn.CrossEntropyLoss()
    n=len(X)

    #shuffle dataset
    for e in range(EPOCHS):
        #random permutation of indicees 0 to n-1
        perm = torch.randperm(n)
        X,Y=X[perm], Y[perm]

        total_loss = 0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            #moves 4each batch to GPU/CPU right before using
            xb = X[i:i+BATCH_SIZE].to(DEVICE)
            yb = Y[i:i+BATCH_SIZE].to(DEVICE)

            #feed batch through transformer and get logits of shape (B,T,3)
            logits=model(xb)
            loss = loss_fn(logits.view(-1,3), yb.view(-1))

            #clears previous gradients
            opt.zero_grad()
            #backprop (comute grad of loss with respect to each paramet in th emodel)
            loss.backward()
            #use grads to update the weights
            opt.step()

            total_loss +=loss.item()
            n_batches+=1

        if(e+1)%5==0:
            print(f"epoch{e+1}/{EPOCHS}")
            print(f"loss:{total_loss/n_batches:.4f}")

        #extract all learned weight matrices as a dictionary
        torch.save(model.state_dict(), "model.pt")
        np.save("sequences.npy", seqs)
        np.save("labels.npy", labels)
        print("DONENE")
        return model, seqs, labels
    
if __name__=="__main__":
    model, seqs, labels = train()
