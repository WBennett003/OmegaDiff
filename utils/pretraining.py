import torch
import torch.nn.functional as F
from dataset import dataset_h5

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from VAE import Encoder, Decoder
from CREP import Reaction_encoder

def reparameterise(mu, sigma):
    std = torch.exp(0.5*sigma)
    eps = torch.randn_like(std)
    return mu + eps*std

def step(seq, rxn, enz_enc, enz_dec, rxn_enc, labels, betas=4):
    sigma, mu = enz_enc(seq)
    z = reparameterise(mu, sigma)
    seq_hat = enz_dec(z)

    rxn_embed = rxn_enc(rxn)
    #CREP duo loss
    mu_rxn = torch.einsum('ij, ik -> ii', mu, rxn)
    mu_rxn = (F.cross_entropy(mu_rxn, labels) + F.cross_entropy(mu_rxn.T, labels)) / 2
    
    sigma_rxn = torch.einsum('ij, ik -> ii', sigma, rxn)
    sigma_rxn = (F.cross_entropy(sigma_rxn, labels) + F.cross_entropy(sigma_rxn.T, labels)) / 2

    #VAE loss
    z = reparameterise(mu, sigma)
    reconstruction_loss = F.cross_entropy(seq_hat.permute(0,2,1), seq, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    VAE_loss = reconstruction_loss + kl_divergence * betas
    loss = (mu_rxn, sigma_rxn, VAE_loss)
    return loss

def pretrain(
    token_size = 23,
    d_model = 256,
    embed_size = 256, 
    sequence_length = 1280,
    hidden_dim = 4096,
    latent_dim = 2048,
    beta = 4,

    chem_size = 10240,

    learning_rate = 1e-3,
    epochs = 3,
    batch_size = 10,
    weight_decay = 1e-3,

    dataset_path = 'datasets/enzyme_data.h5'):
    ds = dataset_h5('datasets/enzyme_data.h5', device)
    data_loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=True)

    enc = Encoder(token_size, embed_size, sequence_length, hidden_dim, latent_dim)
    dec = Decoder(latent_dim, hidden_dim, sequence_length, token_size)
    rxn_enc = Reaction_encoder(chem_size, latent_dim)

    

    CREP_labels = torch.arange(0, batch_size, device=device, dtype=torch.long)
    