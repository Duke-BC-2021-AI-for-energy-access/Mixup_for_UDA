from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from multiprocessing import Pool

class TurbineDataset(Dataset):
    def __init__(self, path_file, num_workers=40, transform=None):
        """
            path_file: path to txt_file that contains a list of images, separated by '\n'
        """
        self.data_root = path_file
        self.image_links = []
        with open(path_file, 'r') as f:
            rows = f.read().split('\n')
            for row in rows:
                if len(row) != 0:
                    self.image_links.append(row)
        
        self.transform = transform
        

    def __len__(self):
        return len(self.image_links)

    def __getitem__(self, index):
        img = Image.open(self.image_links[index])
        if self.transform is not None:
            img = self.transform(img)
        img = np.array(img)
        # img = np.transpose(img, (2, 0, 1)).astype('float32')
        img = img.astype('float32')
        return (img, 0)
