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
from utils.train import Mask_sampler, Mutant_samplers

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
        self.ds = dataset_h5(ds_file, device)
        
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

    #     self.aa_weights = torch.as_tensor([0.31657471, 0.96925363, 0.98297024, 0.98832418, 0.98052505,
    #    0.99623613, 0.98924219, 0.97989413, 0.97366834, 0.99191177,
    #    0.98279196, 0.97022777, 0.98471806, 0.99245229, 0.98858837,
    #    0.98471233, 0.98197448, 0.98253395, 0.99545816, 0.9910603 ,
    #    0.97688197, 0.1, 0.1]) old
        
        self.aa_weights = torch.as_tensor([0.0240, 0.0462, 0.0468, 0.0471, 0.0467, 0.0474, 0.0471, 0.0467, 0.0464,
        0.0472, 0.0468, 0.0462, 0.0469, 0.0473, 0.0471, 0.0469, 0.0468, 0.0468,
        0.0474, 0.0472, 0.0465, 0.0194, 0.0194], device=self.device) # softmaxed

    def train(self, EPOCHS=15, EPOCH_SIZE=10000, BATCH_SIZE=5, lr=1e-3, s=3, wab=False, p=0.5, target_mask=0.15, mask_rate=0.5, verbose_step=50, checkpoint_backprop=True, scaleing=True, sampler='mask', mutate_rate=0.1):
        EPOCH_STEPS = int(EPOCH_SIZE / BATCH_SIZE)
        EPOCH_SIZE = EPOCH_STEPS * BATCH_SIZE
        length = len(self.ds)
        optimizer = torch.optim.AdamW(self.Model.parameters())
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=EPOCHS, steps_per_epoch=EPOCH_STEPS)
        loss_func = torch.nn.CrossEntropyLoss(self.aa_weights)
        
        if sampler == 'mask':
            self.sampler = Mask_sampler(self.sequence_length, BATCH_SIZE, targets=target_mask, mask_rate=mask_rate, device=self.device)
        elif sampler == 'mutant':
            self.sampler = Mutant_samplers(self.sequence_length, BATCH_SIZE, self.token_size, target_mask, mutate_rate, mask_rate, self.device)

        if self.device.type == "cuda" and scaleing: #reduced precision to improve performance https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/#4-use-automatic-mixed-precision-amp-
            print("scaling")
            scaler = torch.cuda.amp.GradScaler()
            torch.backends.cudnn.benchmark = True #improve performance
        elif self.device.type == "cuda:0":
            torch.backends.cudnn.benchmark = True #improve performance


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
            prev_batch = 0
            batch_idx = torch.randperm(self.train_length)[:EPOCH_SIZE]
            for batch in range(1, EPOCH_STEPS):
                optimizer.zero_grad(set_to_none=True) #some witchcraft improvement https://pytorch.org/docs/stable/optim.html
                # get random samples 
                bi, _ = torch.sort(batch_idx[(batch-1)*BATCH_SIZE:batch*BATCH_SIZE])
                Xtokens, rxn = self.ds[bi]
                Xtokens = torch.maximum(z, Xtokens)
                
                #get random 
                x, mask = self.sampler.sample(Xtokens)

                if self.device.type == 'cuda' and scaleing: # Apparently fixed if statement doesn't effect perforance largely?
                    with torch.cuda.amp.autocast(): # speeds up by using f16 instead of f32 https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/ 
                        y_hat = self.Model(x, rxn, mask=mask, fwd_cfg=self.fwd_cfg, s=s)
                        loss = loss_func(y_hat.permute(0, 2, 1), Xtokens)

                else:
                    y_hat = self.Model(x, rxn, mask=mask, fwd_cfg=self.fwd_cfg, s=s)
                    loss = loss_func(y_hat.permute(0, 2, 1), Xtokens)
                

                if self.device.type == 'cuda' and scaleing:
                    scaler.scale(loss).backward() #scaler effecting the gradients
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                loss_sum += loss.detach()
                schedular.step()    
                
                
                if batch % verbose_step == 0:
                    l = ((loss_sum-prev_batch) / verbose_step).item()
                    print(f"{int(100*batch / EPOCH_STEPS)}% | {time.ctime(time.time())} |  loss {l}")
                    prev_batch = loss_sum.item()
                    if wab:
                        wandb.log({
                        "loss" : l,
                        "epoch" : epoch + batch / EPOCH_STEPS, 
                        "lr" : schedular.get_last_lr()[0]
                  })
            
            self.log(epoch, (loss_sum/EPOCH_STEPS).item(), BATCH_SIZE, wab=wab)

    def log(self, epoch, train_loss, batch_size, wab=False):
        self.save_weights(self.model_weight_dir+'_'+str(epoch)+'.pt')
        val_loss, val_acc, seq_txt, pred_seq_txt, accension, distribution_fig = self.evaluate(batch_size)
        

        print(f"Epoch {epoch} : {time.ctime(time.time())} : train_loss {train_loss} : val_loss {val_loss} : val_acc {val_acc}")

        if wab:
            wandb.log(
            {
                "epoch" : epoch,
                "train_loss" : train_loss,
                "val_loss" : val_loss,
                "val_acc" : val_acc,
                "seq_text" : seq_txt[0],
                "pred_seq_text" : pred_seq_txt[0], 
                "accension" : list(accension)[0],
                "distribution" : wandb.Image(distribution_fig)
            }
            )


    def save_weights(self, file_dir):
      torch.save(self.Model.state_dict(), file_dir)

    def load_weights(self, file_dir):
      self.Model.load_state_dict(torch.load(file_dir))

    def evaluate(self, batch_size, eval_size=2, mask=None, guidance=1, show_steps=False, save_dir='denoise.gif'):
        if eval_size > self.val_length:
            eval_size = self.val_length
            
        if eval_size < batch_size:
            eval_size = batch_size

        batch_steps = int(eval_size/batch_size)
        if batch_steps < 1:
            batch_steps = 1
        
        acc_sum = 0
        loss_sum = 0

        val_shuffle = torch.randperm(self.val_length)[:eval_size]
        acc_func = Accuracy(task='multiclass', top_k=1, num_classes=self.token_size).to(self.device)
        loss_func = torch.nn.CrossEntropyLoss(self.aa_weights)
        x = torch.tensor([21], device=self.device, dtype=torch.long).repeat(self.sequence_length).repeat(batch_size).reshape((batch_size, self.sequence_length))
        z = torch.zeros((batch_size,self.sequence_length), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for batch in range(0, batch_steps):
                idx, _ = torch.sort(val_shuffle[batch_size*(batch): batch_size*(batch+1)])
                idx += self.train_length 
                seq, rxn, accension = self.ds.get_row(idx)
                seq = torch.maximum(z, seq)
                x, mask = self.sampler.sample(seq)

                pred_seq = self.Model(x, rxn, fwd_cfg=self.fwd_cfg, s=guidance)
                
                loss = loss_func(pred_seq.permute(0,2,1), seq)
                acc = acc_func(pred_seq.permute(0,2,1).softmax(1), seq)
                acc_sum += acc
                loss_sum += loss

        acc = acc_sum / batch_steps
        loss = loss_sum / batch_steps
        
        seq_txt = [self.tokeniser.token_to_string(i) for i in seq]
        pred_seq_txt = [self.tokeniser.token_to_string(i) for i in pred_seq.softmax(-1).argmax(-1)]

        fig = visualise.plot_AA_confusion(seq.cpu(), pred_seq.softmax(-1).argmax(-1).cpu(), get_figure=True)

        return loss, acc, seq_txt, pred_seq_txt, accension, fig
    

    def evaluate_scratch(self, batch_size, eval_size=2, guidance=1, x_type='random'):
        if eval_size > self.val_length:
            eval_size = self.val_length
            
        if eval_size < batch_size:
            batch_size = eval_size

        batch_steps = int(eval_size/batch_size)
        if batch_steps < 1:
            batch_steps = 1
        
        acc_sum = 0
        loss_sum = 0

        val_shuffle = torch.randperm(self.val_length)[:eval_size]
        acc_func = Accuracy(task='multiclass', top_k=1, num_classes=self.token_size).to(self.device)
        loss_func = torch.nn.CrossEntropyLoss(self.aa_weights)
        if x_type =='random':
            x = torch.randint(low=0, high=22, size=(batch_size, self.sequence_length), dtype=torch.long, device=self.device)
        elif x_type == '*':
            x = torch.tensor([21], device=self.device, dtype=torch.long).repeat(self.sequence_length).repeat(batch_size).reshape((batch_size, self.sequence_length))

        z = torch.zeros((batch_size,self.sequence_length), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            for batch in range(0, batch_steps):
                idx, _ = torch.sort(val_shuffle[batch_size*(batch): batch_size*(batch+1)])
                idx += self.train_length 
                seq, rxn, accension = self.ds.get_row(idx)
                seq = torch.maximum(z, seq)

                pred_seq = self.Model(x, rxn, fwd_cfg=self.fwd_cfg, s=guidance)
                
                loss = loss_func(pred_seq.permute(0,2,1), seq)
                acc = acc_func(pred_seq.permute(0,2,1).softmax(1), seq)
                acc_sum += acc
                loss_sum += loss

        acc = acc_sum / batch_steps
        loss = loss_sum / batch_steps
        
        seq_txt = [self.tokeniser.token_to_string(i) for i in seq]
        pred_seq_txt = [self.tokeniser.token_to_string(i) for i in pred_seq.softmax(-1).argmax(-1)]

        fig = visualise.plot_AA_confusion(seq.cpu(), pred_seq.softmax(-1).argmax(-1).cpu(), get_figure=True)

        return loss, acc, seq_txt, pred_seq_txt, accension, fig
    

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
    runner = Enzyme(token_size=23, chem_size=10240, layers=6, ds_file='10240_2_true_true_500k.h5', embed_weights_file='weights/embed.pt', unbed_weights_file='weights/unbed.pt', model_weight_dir='weights/OmegaDiff_10240')
    runner.Model.load_state_dict(torch.load('weights/OmegaDiff_10240_4.pt'))
    # runner.Model.initalise_plm_weights('release2.pt')
    runner.train(EPOCHS=30, EPOCH_SIZE=2500, BATCH_SIZE=12, lr=1e-3, s=1, wab=False, verbose_step=20, scaleing=False, target_mask=0.15, mask_rate=0.7, sampler='mutant', mutate_rate=0.2)


if __name__ =='__main__':
    train()