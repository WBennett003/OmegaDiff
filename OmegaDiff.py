import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import h5py

import argparse

from omegafold.omegaplm_diffusion import OmegaPLM
from omegafold.config import make_config

class Embedder(nn.Module):
    def __init__(self, token_size=23, d_model=1280, file_name='weights/embed.pt'):
        super().__init__()
        self.embed = nn.Embedding(token_size, d_model)

        self.load_state_dict(torch.load(file_name))
    
    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.embed(x)

class Unbedder(nn.Module):
    def __init__(self, token_size=23, d_model=1280):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, token_size),
            nn.SiLU(),
            nn.Linear(token_size, token_size)
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x.softmax(-1) #give probabilities of each residue

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class Conditioner(torch.nn.Module):
    def __init__(self, t_steps, chem_size=2048, hidden_size=2048, d_model=1280, sequnece_length=1280):
        self.chem_size = chem_size,
        self.sequnece_length = sequnece_length

        self.mask = torch.zeros(sequnece_length)
        self.t_embed = TimestepEmbedder(d_model, t_steps)
        self.reaction_MLP = nn.Sequential(
            nn.Linear(chem_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, d_model),
            nn.SiLU()
        )
        self.mask_MLP = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU()
        )
    
    def null_forward(self, t):
        rxn = torch.zeros((t.shape[0], self.chem_size))
        mask = torch.zeros((t.shape[0], self.sequnece_length))
        x = self.forward(rxn, t, mask)
        return x
    
    def forward(self, rxn, t, mask=None):
        if mask is None:
            mask = self.mask.repeat(t.shape[0]) #repeat for batch_size
        
        t = self.t_embed(t)
        rxn = self.reaction_MLP(rxn)
        mask = self.mask_MLP(mask)
        cond = t + rxn + mask

        return cond

class OmegaDiff(nn.Module):
    def __init__(self, cfg, t_steps=200, chem_size=2048, hidden_size=2048):
        super().__init__()
        
        d_model = cfg.nodes
        sequence = cfg.nodes

        self.omega_plm = OmegaPLM(cfg)
        self.condition = Conditioner(t_steps, chem_size, hidden_size, d_model, sequence)

    def forward(self, xt, t, rxn, mask=None, fwd_cfg=None, s=1):
        if mask is None:
            mask = torch.zeros(xt.shape)

        cond = self.condition(rxn, t, mask)
        xt = self.omega_plm(xt, mask, cond, fwd_cfg)
        xt_null = self.null_forward(xt, t, mask, fwd_cfg)
        xt = xt_null + s * (xt - xt_null)
        return xt
    
    def null_forward(self, xt, t, mask=None, fwd_cfg=None):
        if mask is None:
            mask = torch.zeros(xt.shape)

        cond = self.condition.null_forward(t)
        xt = self.omega_plm(xt, mask, cond, fwd_cfg)
        return xt


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class Diffusion:
    def __init__(self, timesteps):
        self.timesteps = timesteps

        self.betas = self.cosine_beta_schedule()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        posterior_varience = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def cosine_beta_schedule(self, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_samples(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t, t_index, pred_noise):
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)
        
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_varience, t, pred_noise.shape)
            noise = torch.randn_like(pred_noise)
            return model_mean + torch.sqrt(posterior_varience_t) * noise


class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, file_path='datasets/2048_1M.h5py', device='cpu') -> None:
        super().__init__()
        self.file = h5py.File(file_path, 'r+')
        self.device = device
    
    def __len__(self):
        return len(self.file['accension'])

    def __getitem__(self, idx):
        return (
            torch.Tensor(self.file['AA_seq'][idx]).long().to(self.device), 
            torch.Tensor(self.file['reaction'][idx]).to(self.device),
        )
    
    def get_row(self, idx):
        return (
            torch.Tensor(self.file['AA_seq'][idx]).long().to(self.device), 
            torch.Tensor(self.file['reaction'][idx]).to(self.device),
            self.file['smile_reaction']
        
        )

class Enzyme:
    def __init__(self, token_size=23, chem_size=2048, timesteps=200, sequence_length=1280, ds_file='datasets/2048_1M.h5', embed_weight_file='weights/embed.pt'):
        self.device = torch.device('cuda') if torch.cuda.is_avaliable() else torch.device('cpu')
        self.token_size = token_size
        self.chem_size = chem_size
        self.timesteps = timesteps
        self.sequence_length = sequence_length

        self.diffusion = Diffusion(timesteps)
        self.ds = dataset_h5(ds_file)
        
        self.cfg = make_config().plm

        self.d_model = self.cfg.node

        self.fwd_cfg = argparse.Namespace(
        subbatch_size=1000,
        num_recycle=1,
    )   
        self.Embedder = Embedder(self.token_size, self.d_model, embed_weight_file)
        self.Model = OmegaDiff(self.cfg, self.timesteps, self.chem_size, self.chem_size)

    def train(self, EPOCHS=15, EPOCH_SIZE=10000, BATCH_SIZE=5, lr=1e-3, s=3):
        EPOCH_STEPS = int(EPOCH_SIZE / BATCH_SIZE)
        EPOCH_SIZE = EPOCH_STEPS * BATCH_SIZE
        length = len(self.ds)
        optimizer = torch.optim.AdamW(self.Model.parameters())
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, EPOCHS)

        for epoch in range(0, EPOCHS):
            batch_idx = torch.randperm(length)[:EPOCH_SIZE]
            for batch in range(1, EPOCH_STEPS):
                optimizer.zero_grad()
                bi = batch_idx[(batch-1)*BATCH_SIZE:batch*BATCH_SIZE].sort()
                Xtokens, rxn = self.ds[bi]

                x0 = self.Embedder(Xtokens)
                ts = torch.randint(1, self.timesteps, tuple([BATCH_SIZE]))
                noise = torch.randn(x0.shape)
                xt = self.diffusion.q_samples(x0, ts, noise)

                xt = xt.to(self.device)
                ts = ts.to(self.device)
                noise = noise.to(self.device)

                y_hat = self.Model(xt, ts, rxn, fwd_cfg=self.fwd_cfg, s=s)

                loss = F.mse_loss(y_hat, noise)
                loss.backwards()
                optimizer.step()
            
            schedular.step()                

    def sample(self, rxn, timesteps, mask=None, xt=None, guidance=1):
        batch_size = rxn.shape[0]
        if len(rxn.shape) > 1:
            batch_size = rxn.shape[1]
        else:
            batch_size = 1
            rxn = rxn.reshape((batch_size, rxn.shape[0]))

        if xt is None:
             xt = torch.randn((batch_size, self.sequence_length, self.d_model))           
        
        for t in range(1, timesteps)[::-1]:
            t = torch.tensor([t]).repeat(batch_size)
            with torch.no_grad():
                noise = self.Model(xt, t, rxn, mask, self.fwd_cfg, s=guidance)
            xt = self.diffusion.p_sample(xt, t, t, noise)




if __name__ =='__main__':
    runner = Enzyme(token_size=23, chem_size=2048, timesteps=200, ds_file='dataset/test.h5', embed_weights_file='weights/embed.pt')
    runner.train(EPOCHS=15, EPOCH_SIZE=10000, BATCH_SIZE=5, lr=1e-3, s=3)