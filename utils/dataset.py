import torch
import h5py


class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, file_path='datasets/2048_1M.h5py', device='cpu') -> None:
        super().__init__()
        self.file = h5py.File(file_path, 'r+')
        self.device = torch.device(device)
    
    def __len__(self):
        return len(self.file['accension'])

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.file['AA_seq'][idx], device=self.device, dtype=torch.long), 
            torch.as_tensor(self.file['reaction'][idx], device=self.device, dtype=torch.float),
        )
    
    def get_row(self, idx):
        return (
            torch.as_tensor(self.file['AA_seq'][idx], device=self.device, dtype=torch.long), 
            torch.as_tensor(self.file['reaction'][idx], device=self.device, dtype=torch.float),
            self.file['accension']
        )