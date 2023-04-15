import torch
import time
import wandb

import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from torchmetrics import Accuracy

from omegafold.config import make_config

from utils.tokeniser import Tokeniser
from utils.dataset import dataset_h5
from utils.models import OmegaCtrl

import visualise

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Enzyme:
    def __init__(self, token_size=23, chem_size=2048, sequence_length=1280, layers=1, ds_file='datasets/2048_1M.h5', embed_weights_file='weights/embed.pt', unbed_weights_file='weights/unbed.pt', model_weight_dir='weights/OmegaDiff.pt', train_val_test_split=(0.8, 0.1, 0.1)):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.token_size=token_size
        self.chem_size = chem_size
        self.sequence_length = sequence_length
        self.model_weight_dir = model_weight_dir

        self.tokeniser = Tokeniser()
        self.ds = dataset_h5(ds_file)
        
        length = len(self.ds)
        
        self.train_length = int(length*train_val_test_split[0])
        self.val_length = int(length*train_val_test_split[1])
        self.test_length = int(length*train_val_test_split[2])
        
        self.cfg = make_config()
        self.cfg_geo = self.cfg
        self.cfg = self.cfg.plm
        
        self.cfg.edge = layers

        self.d_model = self.cfg.node

        self.fwd_cfg = argparse.Namespace(
        subbatch_size=1280,
        num_recycle=1,
    )   
        self.Model = OmegaCtrl(self.cfg, self.token_size, self.chem_size, self.chem_size, self.cfg_geo)
        self.Model.to(self.device)

    def train(self, EPOCHS=15, EPOCH_SIZE=10000, BATCH_SIZE=5, lr=1e-3, s=3, wab=False, p=0.5):
        EPOCH_STEPS = int(EPOCH_SIZE / BATCH_SIZE)
        EPOCH_SIZE = EPOCH_STEPS * BATCH_SIZE
        length = len(self.ds)
        optimizer = torch.optim.AdamW(self.Model.parameters())
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=EPOCHS, steps_per_epoch=EPOCH_STEPS)
        loss_func = torch.nn.CrossEntropyLoss()

        if wab:
            wandb.init(
            # set the wandb project where this run will be logged
            project="OmegaDiff",
            
            # track hyperparameters and run metadata
            config={
            "type-model" : 'no-diff',
            "learning_rate": lr,
            "dataset": "2048_1M.h5",
            "epochs": EPOCHS,
            "epoch_size": EPOCH_SIZE,
            "batch_size": BATCH_SIZE,
            "layers": self.cfg.edge,
            "chem_size": self.chem_size,
            "guided multiplier": s,
            })

        n_masked = (self.sequence_length*p)

            
        z = torch.zeros((BATCH_SIZE,self.sequence_length), dtype=torch.long, device=self.device)
        x = torch.tensor([21], device=self.device, dtype=torch.long).repeat(self.sequence_length).repeat(BATCH_SIZE).reshape((BATCH_SIZE, self.sequence_length))
        for epoch in range(0, EPOCHS):
            loss_sum = 0
            batch_idx = torch.randperm(self.train_length)[:EPOCH_SIZE]
            for batch in range(1, EPOCH_STEPS):
                optimizer.zero_grad()
                # get random samples 
                bi, _ = torch.sort(batch_idx[(batch-1)*BATCH_SIZE:batch*BATCH_SIZE])
                Xtokens, rxn = self.ds[bi]
                Xtokens = torch.maximum(z, Xtokens)
                
                #get random 

                
                y_hat, pairs = self.Model(x, rxn, fwd_cfg=self.fwd_cfg, s=s)
                loss = loss_func(
                    y_hat.permute(0, 2, 1),
                    Xtokens)

                loss.backward()
                loss_sum += loss.detach()
                optimizer.step()
                schedular.step()    
                
                
                if batch % 25 == 0:
                    l = round((loss_sum / batch).item(), 4)
                    print(f"{int(100*batch / EPOCH_STEPS)}% | {time.ctime(time.time())} |  loss {l}")
                    if wab:
                        wandb.log({
                        "loss" : l,
                        "lr" : schedular.get_last_lr()[0]
                  })
            
            
            self.log(epoch, (loss_sum/EPOCH_STEPS).item(), BATCH_SIZE, wab)

    def log(self, epoch, train_loss, batch_size, wab=False):
        self.save_weights(self.model_weight_dir+'_'+str(epoch)+'.pt')
        val_loss, val_acc, seq_txt, pred_seq_txt, accension, pairs = self.evaluate(batch_size)

        if wab:
            wandb.log(
            {
                "epoch" : epoch,
                "train_loss" : train_loss,
                "val_loss" : val_loss,
                "val_acc" : val_acc,
                "seq_text" : seq_txt,
                "pred_seq_text" : pred_seq_txt, 
                "accension" : accension,
                # "pair_weights" : wandb.Image(pairs)
            }
            )

        print(f"Epoch {epoch} : {time.ctime(time.time())} : train_loss {train_loss} : val_loss {val_loss} : val_acc {val_acc}")

    def save_weights(self, file_dir):
      torch.save(self.Model.state_dict(), file_dir)

    def load_weights(self, file_dir):
      self.Model.load_state_dict(torch.load(file_dir))

    def evaluate(self, batch_size, eval_size=500, mask=None, guidance=1, show_steps=False, save_dir='denoise.gif'):
        if eval_size > self.val_length:
            eval_size = self.val_length

        batch_steps = int(eval_size/batch_size)
        if batch_steps < 1:
            batch_steps = 1
        
        acc_sum = 0
        loss_sum = 0

        val_shuffle = torch.randperm(self.val_length)[:eval_size]
        acc_func = Accuracy(task='multiclass', top_k=1, num_classes=self.token_size).to(self.device)
        loss_func = torch.nn.CrossEntropyLoss()
        x = torch.tensor([21], device=self.device, dtype=torch.long).repeat(self.sequence_length).repeat(batch_size).reshape((batch_size, self.sequence_length))
        z = torch.zeros((batch_size,self.sequence_length), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for batch in range(0, batch_steps):
                idx, _ = torch.sort(val_shuffle[batch_size*(batch): batch_size*(batch+1)])
                idx += self.train_length 
                seq, rxn, accension = self.ds.get_row(idx)
                seq = torch.maximum(z, seq)
                
                pred_seq, pairs_matrix = self.Model(x, rxn, fwd_cfg=self.fwd_cfg, s=guidance)
                
                loss = loss_func(pred_seq.permute(0,2,1), seq)
                acc = acc_func(pred_seq.permute(0,2,1), seq)
                acc_sum += acc
                loss_sum += loss

        print(pairs_matrix.shape)
                
        acc = acc_sum / batch_steps
        loss = loss_sum / batch_steps
        
        seq_txt = [self.tokeniser.token_to_string(i) for i in seq]
        pred_seq_txt = [self.tokeniser.token_to_string(i) for i in pred_seq.argmax(-1)]
        # pair_fig = self.visualise_pairs_matrix(pairs_matrix)
        return loss, acc, seq_txt, pred_seq_txt, accension, None
    
    def visualise_pairs_matrix(self, pair_matrix, figsize=(20,8)):
        bs = pair_matrix.shape[0]
        print(bs)
        fig, ax = plt.subplots(bs, 1, figsize=figsize)

        for i in range(bs):
            sns.heatmap(pair_matrix[i], ax=ax[i, 0])
        return fig

def test_inference():
    runner = Enzyme(token_size=23, chem_size=2048, timesteps=200, layers=6, ds_file='2048_1M.h5', embed_weights_file='OmegaDiff/weights/embed.pt', unbed_weights_file='OmegaDiff/weights/unbed.pt', model_weight_dir='/content/drive/My Drive/OmegaDiff')
    runner.Model.load_state_dict(torch.load('/content/drive/My Drive/OmegaDiff_2.pt'))
    t, p, anime = runner.evaluate([69, 420], guidance=3, show_steps=True)

def train():
    runner = Enzyme(token_size=23, chem_size=10240, layers=6, ds_file='10240_2_true_true_500k.h5', embed_weights_file='OmegaDiff/weights/embed.pt', unbed_weights_file='OmegaDiff/weights/unbed.pt', model_weight_dir='OmegaDiff/weights/OmegaDiff_10240')
    # runner.Model.load_state_dict(torch.load('OmegaDiff_2.pt'))
    runner.Model.initalise_plm_weights('release2.pt')
    runner.train(EPOCHS=30, EPOCH_SIZE=10000, BATCH_SIZE=10, lr=0.1, s=1, wab=False)

if __name__ =='__main__':
    train()