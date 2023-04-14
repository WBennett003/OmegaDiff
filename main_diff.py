import torch
import math
import time
import wandb

import argparse
import matplotlib.pyplot as plt
import seaborn as sns


from omegafold.config import make_config

from utils.diffusion import Diffusion
from utils.tokeniser import Tokeniser
from utils.dataset import dataset_h5
from utils.models import Embedder, Unbedder, OmegaCond

import visualise

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        self.Embedder = Embedder(self.token_size, self.d_model, embed_weights_file).to(self.device)
        self.Unbedder = Unbedder(self.token_size, self.d_model, self.sequence_length, unbed_weights_file).to(self.device)
        self.Model = OmegaCond(self.cfg, self.timesteps, self.chem_size, self.chem_size)
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

        z = torch.zeros((BATCH_SIZE, self.sequence_length)).long().to(self.device)

        for epoch in range(0, EPOCHS):
            loss_sum = 0
            batch_idx = torch.randperm(length)[:EPOCH_SIZE]
            for batch in range(1, EPOCH_STEPS):
                optimizer.zero_grad()
                bi, _ = torch.sort(batch_idx[(batch-1)*BATCH_SIZE:batch*BATCH_SIZE])
                Xtokens, rxn = self.ds[bi]
                Xtokens = torch.maximum(z, Xtokens)

                x0 = self.Embedder(Xtokens)
                ts = torch.randint(1, self.timesteps, tuple([BATCH_SIZE]), device=self.device)
                noise = torch.randn(x0.shape, device=self.device)
                xt = self.diffusion.q_samples(x0, ts, noise)

                # xt = xt.to(self.device)
                # ts = ts.to(self.device)
                # noise = noise.to(self.device)
                # rxn = rxn.to(self.device)

                y_hat = self.Model(xt, ts, rxn, fwd_cfg=self.fwd_cfg, s=s)

                loss = loss_func(y_hat, noise)
                loss.backward()
                loss_sum += loss.detach()
                optimizer.step()
                schedular.step()                    
                if batch % 50 == 0:
                  print(f"{int(100*batch / EPOCH_STEPS)} | {time.ctime(time.time())} |  MSE {round(loss_sum / batch, 4)}")
                  if wab:
                      wandb.log({
                      "loss" : loss_sum / batch,
                      "epoch" : epoch + batch / EPOCH_STEPS 
                  })
            
            self.log(epoch, loss_sum/EPOCH_STEPS, x0.detach().cpu(), xt.detach().cpu(), noise.detach().cpu(), y_hat.detach().cpu(), wab)

    def log(self, epoch, loss, x0, xt, y, y_hat, wab=False):
        self.save_weights(self.model_weight_dir+'_'+str(epoch)+'.pt')
        img = self.visualise_training(x0, xt, y, y_hat)
        seq, pred_seq, anim = self.evaluate([69, 420], guidance=3, show_steps=False)

        if wab:
            wandb.log(
            {
                "epoch" : epoch,
                "loss" : loss,
                "true_seq" : seq,
                "pred_seq" : pred_seq,
                "train_fig" : wandb.Image(img),
                # "denoising" : wandb.Video("denoise.gif")
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
            batch_size = rxn.shape[0]
        else:
            batch_size = 1
            rxn = rxn.reshape((batch_size, rxn.shape[0]))

        if xt is None:
             xt = torch.randn((batch_size, self.sequence_length, self.d_model))           
        
        t = torch.tensor([t]).repeat(batch_size).to(self.device)
        with torch.no_grad():
            noise = self.Model(xt, t, rxn, mask, self.fwd_cfg, s=guidance)
        xt = self.diffusion.p_sample(xt, t, 1, noise)
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
    
    def inference(self, rxn, timesteps, mask=None, guidance=1, batch_size=2, show_steps=False):
        if mask is None:
          mask = torch.zeros((rxn.shape[0], self.sequence_length)).to(self.device)
        xt = torch.randn((batch_size, self.sequence_length, self.d_model)).to(self.device)
        x0, steps = self.sample_loop(rxn, timesteps, mask, xt, guidance, save_steps=show_steps)
        with torch.no_grad():
          self.Unbedder
          tokens = self.Unbedder(x0).argmax(-1)
        aas = [self.tokeniser.token_to_string(tokens[i]) for i in range(tokens.shape[0])]
        return aas, steps  

    def evaluate(self, idxs, mask=None, guidance=1, show_steps=False, save_dir='denoise.gif'):
        seq, rxn = self.ds[idxs]
        rxn = rxn.to(self.device)
        seq = [self.tokeniser.token_to_string(seq[i]) for i in range(seq.shape[0])]
        print(seq)
        pred_seqs, steps = self.inference(rxn, self.timesteps, guidance=guidance, show_steps=show_steps)
        
        if len(steps) > 0:
            animation = visualise.denoising_animation(steps)
            animation.save(save_dir)
        else:
            animation = None
        
        print(f"True : {seq} \n Pred : \n" + '\n'.join(['i'+a for i,a in enumerate(pred_seqs)]))
        return seq, pred_seqs, animation
    
def test_inference():
    runner = Enzyme(token_size=23, chem_size=2048, timesteps=200, layers=6, ds_file='2048_1M.h5', embed_weights_file='OmegaDiff/weights/embed.pt', unbed_weights_file='OmegaDiff/weights/unbed.pt', model_weight_dir='/content/drive/My Drive/OmegaDiff')
    runner.Model.load_state_dict(torch.load('/content/drive/My Drive/OmegaDiff_2.pt'))
    t, p, anime = runner.evaluate([69, 420], guidance=3, show_steps=True)

def train():
    runner = Enzyme(token_size=23, chem_size=10240, timesteps=200, layers=10, ds_file='10240_2_true_true_500k.h5', embed_weights_file='OmegaDiff/weights/embed.pt', unbed_weights_file='OmegaDiff/weights/unbed.pt', model_weight_dir='OmegaDiff/weights/OmegaDiff_10240')
    runner.Model.load_state_dict(torch.load('OmegaDiff_2.pt'))
    runner.Model.initalise_plm_weights('release2.pt')
    runner.train(EPOCHS=30, EPOCH_SIZE=5000, BATCH_SIZE=4, lr=1e-3, s=1, wab=False)

if __name__ =='__main__':
    train()