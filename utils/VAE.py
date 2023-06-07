import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR


class Encoder(nn.Module):
    def __init__(self, token_size, embed_size, sequence_length, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(token_size, embed_size)
        self.flatten = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(sequence_length*embed_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        e1 = self.embed(x)
        e1 = self.flatten(e1)
        h1 = F.silu(self.fc1(e1))
        return self.fc21(h1), self.fc22(h1)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, sequence_length, token_size):
        super(Decoder, self).__init__()

        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, sequence_length*token_size)
        self.unflatten = nn.Unflatten(-1, (sequence_length, token_size))

    def forward(self, z):
        h3 = F.silu(self.fc3(z))
        h3 = self.fc4(h3)
        h3 = self.unflatten(h3)
        return h3
    
class VAE(nn.Module):
    def __init__(self, token_size, embed_size, sequence_length, hidden_dim, latent_dim, betas=1):
        super(VAE, self).__init__()
        self.betas = betas
        self.encoder = Encoder(token_size, embed_size, sequence_length, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, sequence_length, token_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)

        reconstruction_loss = F.cross_entropy(reconstructed_x.permute(0,2,1), x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return reconstruction_loss + kl_divergence * self.betas



def train_vae(model, train_loader, num_epochs, learning_rate=1e-2, weight_decay=1e-5, verbose_n_steps=10, save_path=None):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader))

    model.train()
    least_loss = 1e9

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (inputs, _) in enumerate(train_loader):

            optimizer.zero_grad()
            loss = model(inputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if (i+1) % verbose_n_steps == 0:
                avg_loss = epoch_loss / (i+1)
                print('Epoch [%d/%d], Batch [%d/%d], Avg. Loss: %.4f' % (epoch+1, num_epochs, i+1, len(train_loader), avg_loss))

                if save_path is not None and (i+1) % (verbose_n_steps * 10) == 0 and avg_loss < least_loss:
                    torch.save(model.state_dict(), save_path)
                    least_loss = avg_loss
                    
        avg_epoch_loss = epoch_loss / len(train_loader)
        print('Epoch [%d/%d], Avg. Loss: %.4f' % (epoch+1, num_epochs, avg_epoch_loss))

        if save_path is not None:
            torch.save(model.state_dict(), save_path)

    print('Training finished')


def train():
    from dataset import dataset_h5

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch_size = 10
    epochs = 3
    learning_rate = 5e-4
    weight_decay = 1e-3
    verbose_steps = 20
    save_path='DVAE.pt'
    sequence_length = 1280
    token_size = 23
    embed_size = 256
    input_dim = sequence_length * token_size
    hidden_dim = 10048
    latent_dim = 4096

    betas = 4

    model = VAE(token_size, embed_size, sequence_length, hidden_dim, latent_dim, betas=betas).to(device)
    model.load_state_dict(torch.load('DVAE.pt'))
    ds = dataset_h5('datasets/enzyme_data.h5', device)
    data_loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=True)

    train_vae(model, data_loader, epochs, learning_rate, weight_decay, verbose_steps, save_path)

if __name__ == '__main__':
    train()