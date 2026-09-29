from __future__ import annotations
import torch
from torch import nn

class DriftPFN(nn.Module):
    def __init__(self,input_dim=27,d_model=128,nhead=4,layers=4):
        super().__init__(); self.input_dim=input_dim
        self.embed=nn.Sequential(nn.Linear(input_dim,d_model),nn.GELU(),nn.LayerNorm(d_model))
        enc=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=256,dropout=.1,batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(enc,layers,norm=nn.LayerNorm(d_model),enable_nested_tensor=False)
        self.defect=nn.Linear(d_model,1); self.quantiles=nn.Linear(d_model,9); self.dtype=nn.Linear(d_model,5); self.param=nn.Linear(d_model,3)
    def forward(self,x,padding_mask=None):
        h=self.encoder(self.embed(x),src_key_padding_mask=padding_mask)
        return {'defect':self.defect(h).squeeze(-1),'quantiles':self.quantiles(h).view(*h.shape[:2],3,3),'dtype':self.dtype(h),'param':self.param(h)}
