import torch

class Mutant_samplers:
    def __init__(self, length=1280, batch_size=5, token_size=23, targets=0.15, mutate_rate=0.1, mask_rate=0.8, device=None):
        
        assert mutate_rate + mask_rate < 1 / "to high mutate and mask rates it must be less than 1"

        self.length = length
        self.batch_size = batch_size 
        self.token_size = token_size

        self.tar_length = int(length*targets)
        self.mutate_length = int(self.tar_length*mutate_rate)
        self.mask_length = int(self.tar_length*mask_rate)

        self.mask_aas = torch.as_tensor([21], dtype=torch.long, device=device).repeat(self.mask_length).repeat(self.batch_size).reshape((self.batch_size, self.mask_length))
        self.blank_mask = torch.ones((self.batch_size, self.length), dtype=torch.float, device=device)

    def mutate_sample(self, x):
        assert self.batch_size == x.shape[0]

        for i in range(self.batch_size):
            idxs = torch.randperm(self.length)
            tars = idxs[:self.tar_length]

            mutates, _ = torch.sort(tars[:self.mutate_length])
            mask_idx, _ = torch.sort(tars[self.mutate_length:self.mutate_length+self.mask_length])
            mutant_aas = torch.randint(0, self.token_size-1, (self.batch_size, self.mutate_length), device=x.device)

            x[i, mutates] = mutant_aas
            x[i, mask_idx] = self.mask_aas
            mask = self.blank_mask
            mask[i, mask_idx] = 0.

        return x, mask
    

class Mask_sampler:
    def __init__(self, length=1280, batch_size=5, targets=0.15, mask_rate=0.5, device=None):
        

        self.length = length
        self.batch_size = batch_size 

        self.tar_length = int(length*targets)
        self.mask_length = int(self.tar_length*mask_rate)

        self.indexes = torch.arange(0, self.length, dtype=torch.long, device=device).repeat(self.batch_size).reshape((self.batch_size, self.length))
        
        self.mask_aas = torch.as_tensor([21], dtype=torch.long, device=device).repeat(self.mask_length)
        self.blank_mask = torch.ones((self.batch_size, self.length), dtype=torch.float, device=device)

    def sample(self, x):
        
        mask = self.blank_mask

        for i in range(self.batch_size):
            idxs = torch.randperm(self.length, device=x.device)
            tars = idxs[:self.tar_length]

            mask_idx, _ = torch.sort(tars[:self.mask_length])

            x[i][mask_idx] = self.mask_aas
            mask[i][mask_idx] = 0.

        return x, mask
    
def test_mask_sampler():
    sam = Mask_sampler(1280, 3, 0.15, 0.5)
    a = torch.randint(0, 23, (3, 1280))
    x, mask = sam.sample(a)

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.heatmap(mask)
    plt.show()
    sns.heatmap(x)
    plt.show()

if __name__ =='__main__':
    test_mask_sampler()