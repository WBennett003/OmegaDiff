import torch
import h5py


class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, file_path='datasets/2048_1M.h5py', device='cuda') -> None:
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