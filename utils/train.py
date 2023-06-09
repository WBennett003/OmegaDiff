import torch

MASK_IDX = 22

class Mutant_samplers:
    def __init__(self, length=1280, batch_size=5, token_size=23, targets=0.15, mutate_rate=0.1, mask_rate=0.8, device=None):

        self.length = length
        self.batch_size = batch_size 
        self.token_size = token_size

        self.tar_length = int(length*targets)
        self.mutate_length = int(self.tar_length*mutate_rate)
        self.mask_length = int(self.tar_length*mask_rate)

        self.mask_aas = torch.as_tensor([MASK_IDX], dtype=torch.long, device=device).repeat(self.mask_length)
        self.blank_mask = torch.ones((self.batch_size, self.length), dtype=torch.float, device=device)

    def sample(self, x):
        x = torch.clone(x)
        for i in range(self.batch_size):
            idxs = torch.randperm(self.length)
            tars = idxs[:self.tar_length]

            mutates, _ = torch.sort(tars[:self.mutate_length])
            mask_idx, _ = torch.sort(tars[self.mutate_length:self.mutate_length+self.mask_length])
            mutant_aas = torch.randint(low=0, high=self.token_size-1, size=tuple([self.mutate_length]), device=x.device)

            x[i, mutates] = mutant_aas
            x[i, mask_idx] = self.mask_aas
            mask = self.blank_mask
            mask[i, mask_idx] = 0

        return x, mask
    

class Mask_sampler:
    def __init__(self, length=1280, batch_size=5, targets=0.15, mask_rate=0.5, device=None):

        self.length = length
        self.batch_size = batch_size 

        self.tar_length = int(length*targets)
        self.mask_length = int(self.tar_length*mask_rate)

        self.indexes = torch.arange(0, self.length, dtype=torch.long, device=device).repeat(self.batch_size).reshape((self.batch_size, self.length))
        
        self.mask_aas = torch.as_tensor([MASK_IDX], dtype=torch.long, device=device).repeat(self.mask_length)
        self.blank_mask = torch.ones((self.batch_size, self.length), dtype=torch.float, device=device)

    def sample(self, x):
        
        mask = self.blank_mask
        x = torch.clone(x)

        for i in range(self.batch_size):
            idxs = torch.randperm(self.length, device=x.device)
            tars = idxs[:self.tar_length]

            mask_idx, _ = torch.sort(tars[:self.mask_length])

            x[i][mask_idx] = self.mask_aas
            mask[i][mask_idx] = 0.

        return x, mask
    
# class Half_sampler:
#     def __init__(self, length=1280, batch_size=5, device=None):

#         self.length = length
#         self.batch_size = batch_size 

#         self.mask_aas = torch.as_tensor([MASK_IDX], dtype=torch.long, device=device).repeat(self.mask_length)

#     def sample(self, x):
        
#         idx = x.argmin()

#         return x, mask

class Active_sampler:
    def sample(self, x, active_mask):
        x = torch.clone(x)
        x[active_mask.bool()] = MASK_IDX
        return x

class Scaffold_sampler:
    def sample(self, x, active_mask):
        x = torch.clone(x)
        x[(1-active_mask).bool()] = MASK_IDX
        return x


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

def test_active_sampler():
    a = torch.randint(0, 23, (3, 1280))
    active_mask = torch.zeros_like(a)
    active_mask[:, 5] = 1
    active_mask[:, 8] = 1
    active_mask[:, 10] = 1
    AS = Active_sampler()
    x = AS.scaffold_knockout_sample(a, active_mask)
    x = AS.active_knockout_sample(a, active_mask)
    pass

if __name__ =='__main__':
    test_active_sampler()