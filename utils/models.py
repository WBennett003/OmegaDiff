import torch
import torch.nn as nn
import torch.nn.functional as F

from omegafold.omegaplm_diffusion import OmegaPLM

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Embedder(torch.nn.Module):
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

class OmegaCond(nn.Module):
    def __init__(self, cfg, t_steps=200, chem_size=2048, hidden_size=2048):
        super().__init__()
        
        d_model = cfg.node
        sequence = cfg.node

        self.omega_plm = OmegaPLM(cfg)
        self.condition = Conditioner(t_steps, chem_size, hidden_size, d_model, sequence)

    def initalise_plm_weights(self, w_dir='weights/model2.pt'):
        w = torch.load(w_dir)
        plm = self.omega_plm.state_dict()

        new_state_dict = {}

        for k in plm.keys():
            new_state_dict[k] = plm[k]

        for k in plm.keys():
            if k[:10] == 'omega_plm.':
                new_state_dict[k[10:]] = w[k]
        

        self.omega_plm.load_state_dict(new_state_dict)

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
