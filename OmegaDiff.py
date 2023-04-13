import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import h5py
import wandb

import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from omegafold.omegaplm_diffusion import OmegaPLM
from omegafold.config import make_config

import visualise

class EMBEDDER(torch.nn.Module):
  def __init__(self, token_size, d_model, file_name=''):
    super().__init__()
    self.layer = torch.nn.Embedding(token_size, d_model)
    self.load_state_dict(torch.load(file_name))
    self.freeze_weights()

  def forward(self, x):
    return self.layer(x)

  def freeze_weights(self):
      for param in self.parameters():
          param.requires_grad = False


class Unbedder(nn.Module):
    def __init__(self, token_size=23, d_model=1280, sequence_length=1280, unbed_weight_file=''):
        super().__init__()
        self.token_size = token_size
        self.d_model = d_model
        self.sequence_length = sequence_length

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, token_size),
            nn.SiLU(),
            nn.Linear(token_size, token_size)
        )

        self.load_state_dict(torch.load(unbed_weight_file))
        
    def forward(self, x):
        x = self.mlp(x)
        return x #give probabilities of each residue

    def train(self, ds, EPOCHS, EPOCH_SIZE, BATCH_SIZE, embedder, lr=1e-3):
        EPOCH_STEPS = int(EPOCH_SIZE / BATCH_SIZE)
        EPOCH_SIZE = EPOCH_STEPS * BATCH_SIZE
        length = len(ds)
        optimizer = torch.optim.AdamW(self.parameters())
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, EPOCHS)
        loss_func = torch.nn.MSELoss()

        z = torch.zeros((BATCH_SIZE, self.sequence_length)).long()

        for epoch in range(0, EPOCHS):
            loss_sum = 0
            batch_idx = torch.randperm(length)[:EPOCH_SIZE]
            for batch in range(1, EPOCH_STEPS):
                optimizer.zero_grad()
                bi, _ = torch.sort(batch_idx[(batch-1)*BATCH_SIZE:batch*BATCH_SIZE])
                Xtokens, rxn = ds[bi]
                Xtokens = torch.maximum(z, Xtokens)

                x0 = embedder(Xtokens)

                y_hat = self.forward(x0)
                loss = loss_func(y_hat, Xtokens)
                loss.backward()
                loss_sum += loss.detach()
                optimizer.step()
                if batch % 50 == 0:
                  print(batch / EPOCH_STEPS, loss_sum / batch)
            print(epoch, loss_sum / EPOCH_STEPS)            
            schedular.step()     

        


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
        super().__init__()
        self.chem_size = chem_size
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
        rxn = torch.zeros((t.shape[0], self.chem_size)).to(t.device)
        mask = torch.zeros((t.shape[0], self.sequnece_length)).to(t.device)
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
        
        d_model = cfg.node
        sequence = cfg.node

        self.omega_plm = OmegaPLM(cfg)
        self.condition = Conditioner(t_steps, chem_size, hidden_size, d_model, sequence)

    def forward(self, xt, t, rxn, mask=None, fwd_cfg=None, s=1):
        if mask is None:
            mask = torch.zeros(xt.shape[:2]).to(xt.device)

        cond = self.condition(rxn, t, mask)
        ei = self.omega_plm(xt, mask, cond, fwd_cfg)
        ej = self.null_forward(xt, t, mask, fwd_cfg)
        eh = ej + s * (ei - ej)
        return eh
    
    def null_forward(self, xt, t, mask=None, fwd_cfg=None):
        if mask is None:
            mask = torch.zeros(xt.shape[:2])

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
        
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_varience, t, pred_noise.shape)
            noise = torch.randn_like(pred_noise)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


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

class Tokeniser:
  def __init__(self):
    self.AA = {
    "A" : 0,
    "R" : 1,
    "N" : 2,
    "D" : 3,
    "C" : 4,
    "Q" : 5,
    "E" : 6,
    "G" : 7,
    "H" : 8,
    "I" : 9,
    "L" : 10,
    "K" : 11,
    "M" : 12,
    "F" : 13,
    "P" : 14,
    "S" : 15,
    "T" : 16,
    "W" : 17,
    "Y" : 18,
    "V" : 19,
    "*" : 20,
    "-" : 21
    }
    self.inverse_AA = {
      0 : "A",
      1 : "R",
      2 : "N",
      3 : "D",
      4 : "C",
      5 : "Q",
      6 : "E",
      7 : "G",
      8 : "H",
      9 : "I",
      10 : "L",
      11 : "K",
      12 : "M",
      13 : "F",
      14 : "P",
      15 : "S",
      16 : "T",
      17 : "W",
      18 : "Y",
      19 : "V",
      20 : "*",
      21 : "-",
      22 : "?",
    }

    def token_to_string(self, tokens):
      aa = ''
      for t in tokens:
        aa += self.inverse_AA[t]
      return aa

    def string_to_token(self, string):
      aa = []
      for char in string:
        aa.append(self.AA[char])
      return aa

    

class Enzyme:
    def __init__(self, token_size=23, chem_size=2048, timesteps=200, sequence_length=1280, layers=1, ds_file='datasets/2048_1M.h5', embed_weights_file='weights/embed.pt', unbed_weights_file='weights/unbed.pt', model_weight_dir='weights/OmegaDiff.pt'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.token_size = token_size
        self.chem_size = chem_size
        self.timesteps = timesteps
        self.sequence_length = sequence_length
        self.model_weight_dir = model_weight_dir

        self.diffusion = Diffusion(timesteps)
        self.tokeniser = Tokeniser()
        self.ds = dataset_h5(ds_file)
        
        self.cfg = make_config().plm
        self.cfg.edge = layers

        self.d_model = self.cfg.node

        self.fwd_cfg = argparse.Namespace(
        subbatch_size=1280,
        num_recycle=1,
    )   
        self.Embedder = EMBEDDER(self.token_size, self.d_model, embed_weights_file)
        self.Unbedder = Unbedder(self.token_size, self.d_model, self.sequence_length, unbed_weights_file)
        self.Model = OmegaDiff(self.cfg, self.timesteps, self.chem_size, self.chem_size)
        self.Model.to(self.device)
        self.Model.condition.to(self.device)
        self.Model.omega_plm.to(self.device)

    def train(self, EPOCHS=15, EPOCH_SIZE=10000, BATCH_SIZE=5, lr=1e-3, s=3, wab=False):
        EPOCH_STEPS = int(EPOCH_SIZE / BATCH_SIZE)
        EPOCH_SIZE = EPOCH_STEPS * BATCH_SIZE
        length = len(self.ds)
        optimizer = torch.optim.AdamW(self.Model.parameters())
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, EPOCHS)
        loss_func = torch.nn.MSELoss()

        if wab:
          wandb.init(
            # set the wandb project where this run will be logged
            project="OmegaDiff",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "dataset": "2048_1M.h5",
            "epochs": EPOCHS,
            "epoch_size": EPOCH_SIZE,
            "batch_size": BATCH_SIZE,
            "layers": self.cfg.edge,
            "chem_size": self.chem_size,
            "guided multiplier": s,
            })

        z = torch.zeros((BATCH_SIZE, self.sequence_length)).long()

        for epoch in range(0, EPOCHS):
            loss_sum = 0
            batch_idx = torch.randperm(length)[:EPOCH_SIZE]
            for batch in range(1, EPOCH_STEPS):
                optimizer.zero_grad()
                bi, _ = torch.sort(batch_idx[(batch-1)*BATCH_SIZE:batch*BATCH_SIZE])
                Xtokens, rxn = self.ds[bi]
                Xtokens = torch.maximum(z, Xtokens)

                x0 = self.Embedder(Xtokens)
                ts = torch.randint(1, self.timesteps, tuple([BATCH_SIZE]))
                noise = torch.randn(x0.shape)
                xt = self.diffusion.q_samples(x0, ts, noise)

                xt = xt.to(self.device)
                ts = ts.to(self.device)
                noise = noise.to(self.device)
                rxn = rxn.to(self.device)

                y_hat = self.Model(xt, ts, rxn, fwd_cfg=self.fwd_cfg, s=s)

                loss = loss_func(y_hat, noise)
                loss.backward()
                loss_sum += loss.detach()
                optimizer.step()
                if batch % 50 == 0:
                  print(batch / EPOCH_STEPS, loss_sum / batch)
            
            self.log(epoch, loss_sum/EPOCH_STEPS, x0.detach().cpu(), xt.detach().cpu(), noise.detach().cpu(), y_hat.detach().cpu(), wab)
            schedular.step()                

    def log(self, epoch, loss, x0, xt, y, y_hat, wab=False):
        self.save_weights(self.model_weight_dir+'_'+str(epoch)+'.pt')
        img = self.visualise_training(x0, xt, y, y_hat)
        seq, pred_seq, anim = self.evaluate([69, 420], guidance=10, show_steps=True)

        if wab:
            wandb.log(
            {
                "epoch" : epoch,
                "loss" : loss,
                "true_seq" : seq,
                "pred_seq" : pred_seq,
                "train_fig" : wandb.Image(img),
                "denoising" : wandb.Video(anim)
            }
            )

        print(f"Epoch {epoch} : MSE {loss}")


    def visualise_training(self, x0, xt, y, y_hat):
      err = (y - y_hat)**2
      bs = 2 
      cols = ['X0' , 'Xt', 'Y', 'Yh', 'Y-Yh']

      fig, ax = plt.subplots(bs, 5, figsize=(15,5))
      
      for a, col in zip(ax[0], cols):
          a.set_title(col)

      for i in range(bs):
        sns.heatmap(x0[i], ax=ax[i, 0])
        sns.heatmap(xt[i], ax=ax[i, 1])
        sns.heatmap(y[i], ax=ax[i, 2])
        sns.heatmap(y_hat[i], ax=ax[i, 3])
        sns.heatmap(err[i], ax=ax[i, 4])

      
      fig.tight_layout()
      plt.close()
      return fig

    def save_weights(self, file_dir):
      torch.save(self.Model.state_dict(), file_dir)

    def load_weights(self, file_dir):
      self.Model.load_state_dict(torch.load(file_dir))

    def sample(self, rxn, t, mask=None, xt=None, guidance=1):
        batch_size = rxn.shape[0]
        if len(rxn.shape) > 1:
            batch_size = rxn.shape[1]
        else:
            batch_size = 1
            rxn = rxn.reshape((batch_size, rxn.shape[0]))

        if xt is None:
             xt = torch.randn((batch_size, self.sequence_length, self.d_model))           
        
        t = torch.tensor([t]).repeat(batch_size)
        with torch.no_grad():
            noise = self.Model(xt, t, rxn, mask, self.fwd_cfg, s=guidance)
        xt = self.diffusion.p_sample(xt, t, t, noise)
        return xt
    
    def sample_loop(self, rxn, t, mask=None, xt=None, guidance=1, save_steps=None):
        if xt is None:
            xt = torch.randn((rxn.shape[0], self.sequence_length, self.d_model)).to(self.device)
        steps = []
        for ts in range(0,t):
            if save_steps:
                steps.append(xt)
            xt = self.sample(rxn, ts, mask, xt, guidance)
        return xt, steps
    
    def x0_to_seq(self, x0):
        tokens = self.Unbedder(x0)
        aas = [self.tokeniser.token_to_string(tokens[i]) for i in range(tokens.shape[0])]
        return aas  
    
    def inference(self, rxn, timesteps, mask=None, guidance=1, batch_size=2):
        xt = torch.randn((batch_size, self.sequence_length, self.d_model)).to(self.device)
        x0, _ = self.sample_loop(rxn, timesteps, mask, xt, guidance)
        tokens = self.Unbedder(x0)
        aas = [self.tokeniser.token_to_string(tokens[i]) for i in range(tokens.shape[0])]
        return aas  

    def evaluate(self, idxs, mask=None, guidance=1, show_steps=False, save_dir='denoise.gif'):
        seq, rxn = self.ds[idxs]
        seq = [self.tokeniser.token_to_string(seq[i]) for i in range(seq.shape[0])]
        pred_seqs, steps = self.inference(rxn, self.timesteps, guidance=guidance, show_steps=show_steps)
        
        if len(steps) > 0:
            animation = visualise.denoising_animation(steps)
            animation.save(save_dir)
        else:
            animation = None
        
        print(f"True : {seq} \n Pred : \n" + '\n'.join(['i'+a for i,a in enumerate(pred_seqs)]))
        return seq, pred_seqs, animation
    
def test_inference():
    runner = Enzyme(token_size=23, chem_size=2048, timesteps=200, layers=6, ds_file='2048_1M.h5', embed_weights_file='OmegaDiff/weights/embed.pt', unbed_weights_file='OmegaDiff/weights.unbed.pt', model_weight_dir='/content/drive/My Drive/OmegaDiff')
    runner.Model.load_state_dict(torch.load('/content/drive/My Drive/OmegaDiff_2.pt'))
    t, p, anime = runner.evaluate([69, 420], guidence=3, show_steps=True)

def train():
    runner = Enzyme(token_size=23, chem_size=2048, timesteps=200, layers=6, ds_file='2048_1M.h5', embed_weights_file='OmegaDiff/weights/embed.pt', unbed_weights_file='OmegaDiff/weights.unbed.pt', model_weight_dir='/content/drive/My Drive/OmegaDiff')
    runner.Model.load_state_dict(torch.load('/content/drive/My Drive/OmegaDiff_2.pt'))
    runner.train(EPOCHS=15, EPOCH_SIZE=5000, BATCH_SIZE=5, lr=1e-1, s=3, wab=True)

if __name__ =='__main__':
    test_inference()