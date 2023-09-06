from torch import nn
import torch
import math 
import torch.nn.functional as F

def replicate(layer,N):
    return nn.ModuleList([layer for _ in range(N)])

class Encode(nn.Module):

    def __init__(self,layer,N):
        super(Encode, self).__init__()
        self.layer=replicate(layer,N)
        self.N=N

    def forward(self,X,mask):
        for layer in self.layer:
            X=layer(X,mask)
        return X
    
class LayerNorm(nn.Module):

    def __init__(self,features,eps=1e-16):
        super(LayerNorm, self).__init__()
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))
        self.eps=eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class AddNorm(nn.Module):
    def __init__(self, size, dropout_rate, eps=1e-6):
        super(AddNorm, self).__init__()
        
        # Layer normalization component
        self.norm = LayerNorm(size, eps=eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer_output):
        "Apply residual connection followed by layer normalization"
        # Residual connection
        added_output = x + self.dropout(sublayer_output)
        
        # Layer normalization
        return self.norm(added_output)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: the number of expected features in the input (required).
        d_ff: the number of features of the feedforward network model.
        dropout: the dropout value (default=0.1).
        """
        super(FeedForward, self).__init__()
        
        # Two linear layers with a ReLU activation in between
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key,value,mask=None):
        batch_size = query.size(0)

        # Linear layers
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.d_k**0.5
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)

        # Concatenate heads and pass through final linear layer
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_k * self.num_heads)
        output = self.fc_out(attention_output)

        return output
    
class Encoder(nn.Module):
    def __init__(self, size, FeedForward, Multi_Head_Attention,AddNorm):
        super(Encoder, self).__init__()
        self.self_attn = Multi_Head_Attention
        self.feed_forward = FeedForward
        self.AddNorm=AddNorm
        self.size = size

    def forward(self,x,mask):
        x=self.AddNorm(x, self.self_attn(x,x,x,mask))
        x=self.AddNorm(x, self.feed_forward(x))
        return x  
    
class Embedding(nn.Module):
    def __init__(self, d_model,vocab):
        super(Embedding, self).__init__()
        self.embedding=nn.Embedding(vocab,d_model)

    def forward(self,x):
        return self.embedding(x)   
    
class Positional_Encoding(nn.Module):
    def __init__(self, d_model):
        super(Positional_Encoding, self).__init__()
        self.dmodel=d_model
        
    def forward(self,x):
        n=x.shape[0]
        div_term = torch.exp(torch.arange(0., self.dmodel, 2) * -(math.log(10000.0) / self.dmodel))

        positions = torch.arange(n).unsqueeze(1).float()
        div_term = div_term.unsqueeze(0)
        sin_vals = torch.sin(positions * div_term)
        cos_vals = torch.cos(positions * div_term)

        ZZ = torch.empty(n, self.dmodel)
        ZZ[:, 0::2] = sin_vals
        ZZ[:, 1::2] = cos_vals
        return x+ZZ
    
class Generator(nn.Module):
    def __init__(self,d_model,tgt_vocab):
        super(Generator, self).__init__()
        self.linear=nn.Linear(d_model,tgt_vocab)

    def forward(self,x):
        x=self.linear(x)
        return F.log_softmax(x, dim=-1)
    

class Encoder_Decoder(nn.Module):
    def __init__(self, Encode,Decode,src_embed, tgt_embed, generator):
        super(Encoder_Decoder, self).__init__()
        self.Encode=Encode
        self.Decode=Decode
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.generator=generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.Encode(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.Decode(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class Decode(nn.Module):

    def __init__(self,layer,N):
        super(Decode, self).__init__()
        self.layer=replicate(layer,N)
        self.N=N

    def forward(self,X,y,src_mask,tgt_mask):
        for layer in self.layer:
            X=layer(X,y,src_mask,tgt_mask)
        return X
    
class Decoder(nn.Module):
    def __init__(self, size, FeedForward, Self_Multi_Head_Attention,Encoder_Multi_Head_Attention,AddNorm):
        super(Decoder, self).__init__()
        self.self_attn = Self_Multi_Head_Attention
        self.feed_forward = FeedForward
        self.encoder_attention=Encoder_Multi_Head_Attention
        self.AddNorm=AddNorm
        self.size = size

    def forward(self,x,m,src_mask,tgt_mask):
        x=self.AddNorm(x, self.self_attn(x,x,x,tgt_mask))
        x=self.AddNorm(x, self.encoder_attention(x,m,m,src_mask))
        x=self.AddNorm(x, self.feed_forward(x))
        return x   
    
def make_model(src_vocab,tgt_vocab,N=6,d_model=512,h=8,dropout=0.1,d_ff=2048):
    
    model=Encoder_Decoder(
        Encode(Encoder(d_model,FeedForward(d_model, d_ff, dropout=dropout), MultiHeadAttention(d_model, h),AddNorm(d_model,dropout, eps=1e-6)),N),
        Decode(Decoder(d_model, FeedForward(d_model, d_ff, dropout=dropout), MultiHeadAttention(d_model, h),MultiHeadAttention(d_model,h),AddNorm(d_model,dropout, eps=1e-6)),N),
        nn.Sequential(Embedding(d_model,src_vocab),Positional_Encoding(d_model)),
        nn.Sequential(Embedding(d_model,tgt_vocab),Positional_Encoding(d_model)),
        Generator(d_model,tgt_vocab)
    )
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model

class Batch:
    def __init__(self,src,tgt,pad=2):
        self.src=src
        self.src_mask=(src!=pad).unsqueeze(-2)
        self.tgt=tgt[:,:-1]
        self.tgt_y=tgt[:,1:]
        self.tgt_mask=self.make_std_mask(self.tgt,pad)

    @staticmethod
    def make_std_mask(tgt,pad):
        tgt_mask=(tgt!=pad).unsqueeze(-2)
        tgt_mask=tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask 
    
def data_gen(V,batch_size,n_batches):
    
    for i in range(n_batches):
        # Get the data and labels for the current batch
        data=torch.randint(1,V,size=(batch_size,10))
        data[:,0]=1
        src=data.requires_grad_(False).clone().detach()
        tgt=data.requires_grad_(False).clone().detach()
        yield Batch(src,tgt,pad=0)

test_model = make_model(11, 11, 2)
for batch in data_gen(11,1,1):
    out=test_model.forward(batch.src,batch.tgt,batch.src_mask,batch.tgt_mask)
out=test_model.generator(out)