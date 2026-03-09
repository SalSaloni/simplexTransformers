import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model%n_heads==0

        self.d_model=d_model
        self.n_heads = n_heads
        self.d_head = d_model//n_heads

        #Big matrix with queries, keys, and values
        self.W_in = nn.Linear(d_model, 3*d_model, bias = False)
        #projects the attention result out
        self.W_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.W_in(x)
        q,k,v = qkv.split(self.d_model, dim=-1)

        #splits d_model dimension across the headsd. d_model=64, n_headsd=2 so 64/2=32 dimensions
        def reshape(t):
            return t.view(B,T, self.n_heads, self.d_head).transpose(1,2)
        
        q,k,v = reshape(q), reshape(k), reshape(v)

        scale = math.sqrt(self.d_head)

        #TxT matrix of scores, computes how much every token should attend to other tokens.
        attn = (q@k.transpose(-2,-1))/scale

        #WE want upper triangle so only future positions
        #Not bidirectional encoder so token at pos x can only attend positions 0...x.
        mask = torch.triu(torch.ones(T, T, device = x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)


        #attention weights needed to take a weighted sum of vals. 
        out = attn@v
        out = out.transpose(1,2).contigious().view(B,T,D)
        return self.W_out(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff= nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),nn.Linear(d_ff, d_model))

    def forward(self, x):
        #Adding back x, adding what's already there.
        #ln1, ln2 layernorms means vectors are normalized before attention and feedforwad stabilizing training.
        x=x+self.attn(self.ln1(x))
        #feedfroward layer. 2 linear layers with GELU attention
        x=x+self.ff(self.ln2(x))
        return x
    
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=3, d_model=64, n_heads=2, n_layers=2, context_len=16, d_ff=128):
        super().__init__()
        self.d_model = d_model
        self.context_len = context_len
        self.n_layers = n_layers

        #embeddings are optimized during training 
        #token_emb maps token_ids(0,1,2) to a d_model dimensional vector
        self.token_emb = nn.Embedding(vocab_size, d_model)
        #pos_emb maps each position (0to 15) to a vector
        self.pos_emb = nn.Embedding(context_len, d_model)


        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff)
        for _ in range(n_layers)])

        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias = False)

        self.residuals = {}
        self.store_residuals = False

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device = x.device)

        #INFO
        res = self.token_emb(x)+self.pos_emb(positions)

        #if store_residuals is true then afer eachlayer, save current residual stream
        if self.store_residuals:
            self.residuals[0]=res.detach().cpu()

        for i, block in enumerate(self.blocks):
            res = block(res)

            if self.store_residuals:
                #each residual layer is after the previous, so res[0] after embedding, res[1] after layer 1
                self.residuals[i+1] = res.detach().cpu()

        res = self.ln_final(res)
        #cross entropy loss turns to a probability distrubiton to compare to the next token
        logits = self.unembed(res)

        return logits
    
    def get_residual_stream(self, x):
        self.store_residuals = True
        with torch.no_grad():
            self.forward(x)
        
        self.store_residuals = False
        return self.residuals
        
