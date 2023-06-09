# Emulates image-text CLIP encodings but for reaction enzyme clips
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time 


class Enzyme_encoder(nn.Module): # do you add position encoding?
    def __init__(self, token_size=23, d_model=1280, cofactors = {}) -> None:
        super().__init__() 
        
        self.embed = nn.Embedding(token_size, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        if cofactors != {}:
            self.cofactor = True
            self.cofactor_token_size = cofactors['token_size']
            self.n_cofactor = cofactors['n_tokens']

            self.cofactor_embedder = nn.Embedding(self.cofactor_token_size, d_model)
        else:
            self.cofactor = False


    def forward(self, x, c=None):
        x = self.embed(x)
        if self.cofactor:
            c = self.cofactor_embedder(c)
            x = torch.concat([x, c], axis=1)
        x = self.mlp(x)
        return x

class Reaction_encoder(nn.Module):
    def __init__(self, chem_size=10240, d_model=1280) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(chem_size, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x):
        x = self.mlp(x)
        return x

class CREP(nn.Module):
    def __init__(self, aa_size=23, chem_size=10240, d_model=1280, cofactor = {}) -> None:
        super().__init__()

        self.enzyme_enc = Enzyme_encoder(aa_size, d_model, cofactor)
        self.reaction_enc = Reaction_encoder(chem_size, d_model)

        self.norm_e = nn.LayerNorm(d_model)
        self.norm_r = nn.LayerNorm(d_model)
        
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        pass

    def forward(self, enz_z, react_z):
        enz_z = self.enzyme_enc_forward(enz_z)
        react_z = self.reaction_enc_forward(react_z)

        enz_z = self.norm_e(enz_z)
        react_z = self.norm_r(react_z).unsqueeze(1) # [b, e] -> [b, None, e]

        logits = torch.einsum('ijk, blk -> ib', enz_z, react_z) #* torch.exp(self.logit_scale)
        # logits = logits * torch.exp(self.logit_scale)# [b, s, e] . [b, None, e] -> [b, b]
        return logits

    def enzyme_enc_forward(self, enz_z):
        enz_z = self.enzyme_enc(enz_z) # [n, s, d] -> [n, d_e]
        enz_z = self.norm_e(enz_z)
        return enz_z


    def reaction_enc_forward(self, react_z):
        react_z = self.reaction_enc(react_z)
        react_z = self.norm_r(react_z)
        return react_z


    def train(self, enz, react, labels):
        logits = self.forward(enz, react)

        loss_e = F.cross_entropy(logits, labels)
        loss_r = F.cross_entropy(logits.T, labels)

        loss = (loss_e + loss_r) / 2
        return loss
    

def train(file_path='datasets/enzyme_data.h5', batch_size=10, epoch_size=500000, epochs=2, lr=1.2, sequence_length=1280, verbose=50):
    from dataset import dataset_h5
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    ds = dataset_h5(file_path, device=device)
    data_loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    length = len(ds)
    epoch_size = length #epoch_size * batch_size
    model = CREP(aa_size=23, chem_size=10240, d_model=1280).to(device)
    model.load_state_dict(torch.load('clip_model.pt'))
    EPOCH_STEPS = int(epoch_size / batch_size)

    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(torch.load('clip_optim.pt'))
    schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=EPOCH_STEPS)

    torch.backends.cudnn.benchmark = True
    # scaler = torch.cuda.amp.GradScaler()

    labels = torch.arange(batch_size, device=device)

    z = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=device)
    print(f"starting training on {device.type}, bs {batch_size} with {epoch_size*epochs} steps, and lr {lr}!")
    for epoch in range(epochs):
        loss_sum = 0
        prev_batch = 0
    #     batch_idx = torch.randperm(length)[:EPOCH_SIZE]
        # for batch in range(1, EPOCH_STEPS):
        for batch, (Xtokens, rxn) in enumerate(data_loader):
            optimizer.zero_grad(set_to_none=True) #some witchcraft improvement https://pytorch.org/docs/stable/optim.html
            Xtokens = torch.maximum(z, Xtokens)


            # with torch.cuda.amp.autocast():
            #     loss = model.train(Xtokens, rxn, labels)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            loss = model.train(Xtokens, rxn, labels)
            loss.backward()
            optimizer.step()


            loss_sum += loss.detach()

            if batch % verbose == 0:
                l = ((loss_sum-prev_batch) / verbose).item()
                print(f"{int(100*batch / EPOCH_STEPS)}% | {time.ctime(time.time())} |  loss {l} | last loss {loss.detach().item()}" )
                prev_batch = loss_sum.item()
                torch.save(model.state_dict(), 'clip_model.pt')#
                torch.save(optimizer.state_dict(), 'clip_optim.pt')

        schedular.step()
        print(f"Epoch {epoch} ended with {loss_sum.item()/batch+1}")

if __name__ == "__main__":
    train(file_path='datasets/enzyme_data.h5')