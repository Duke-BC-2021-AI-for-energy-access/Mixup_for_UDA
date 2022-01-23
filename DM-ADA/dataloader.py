from torch.utils.data import Dataset
from PIL import Image

class TurbineDataset(Dataset):
    def __init__(self, path_file, transform=None):
        """
            path_file: path to txt_file that contains a list of images, separated by '\n'
        """
        self.data_root = path_file
        self.images = []
        with open(path_file, 'r') as f:
            rows = f.read().split('\n')
            for row in rows:
                self.images.append(Image.open(row))
        
        if transform is not None:
            self.images = transform(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]