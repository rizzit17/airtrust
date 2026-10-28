import torch
from torch import nn
import numpy as np

class SpatialGraph(nn.Module):
    """Inductive star-graph message passing; target node has coordinates, no reading."""
    def __init__(self):
        super().__init__()
        self.message=nn.Sequential(nn.Linear(8,24),nn.ReLU(),nn.Linear(24,16),nn.ReLU())
        self.attention=nn.Linear(16,1)
        self.head=nn.Sequential(nn.Linear(20,24),nn.ReLU(),nn.Linear(24,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,context):
        msg=self.message(x)
        weights=torch.softmax(self.attention(msg).squeeze(-1)-torch.linalg.norm(x[:,:,-2:],dim=-1),dim=1)
        pooled=(msg*weights[:,:,None]).sum(1)
        base=(x[:,:,5]*weights).sum(1)
        return base+self.head(torch.cat((pooled,context),dim=1)).squeeze(-1)

class TemporalLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(1,16,batch_first=True)
        self.head=nn.Sequential(nn.Linear(20,16),nn.ReLU(),nn.Linear(16,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,context):
        _,(h,_)=self.lstm(x[:,:,None])
        return x[:,-1]+self.head(torch.cat((h[-1],context),dim=1)).squeeze(-1)

def aggregate(states,weights,method='mean',f=1):
    w=np.asarray(weights,dtype=float)
    if np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0: raise ValueError('Invalid aggregation weights')
    w=w/w.sum()
    if method=='krum':
        if len(states)<=2*f+2: raise ValueError('Krum requires K > 2f + 2')
        flat=torch.stack([torch.cat([v.flatten() for v in s.values()]) for s in states])
        dist=torch.cdist(flat,flat)**2
        score=dist.sort(dim=1).values[:,1:len(states)-f-1].sum(1)
        return {k:v.clone() for k,v in states[int(score.argmin())].items()}
    out={}
    for key in states[0]:
        x=torch.stack([s[key] for s in states])
        if method=='median': out[key]=torch.quantile(x,.5,dim=0)
        elif method=='trimmed': out[key]=x.sort(dim=0).values[1:-1].mean(0)
        else: out[key]=(x*torch.tensor(w,dtype=x.dtype).reshape((-1,)+(1,)*(x.ndim-1))).sum(0)
    return out
