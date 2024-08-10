import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
import cv2
import mmengine
    
class Dataset(Data.Dataset):
    def __init__(self, root_dir, split_file, suffix, da):
        # init
        self.dir_names = mmengine.list_from_file(split_file)
        self.files = [os.path.join(root_dir, dir_name, dir_name + suffix) for dir_name in self.dir_names]
        assert da in ['wbs', 'chexray', 'brats']
        self.da=da

    def __len__(self):
        # Returns the size of the dataset
        return len(self.files)

    def __getitem__(self, index):
        # Indexes a particular piece of data in the dataset and can also pre-process the data
        # The subscript index parameter is mandatory and has an arbitrary name.
        if self.da == 'brats':
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))
            # Fill 0 to (160, 240, 240) size
            img_arr = np.pad(img_arr, ((2, 3), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))[np.newaxis, ...]
            # The return value is automatically converted to the torch's tensor type
        else:
            img_arr = cv2.imread(self.files[index], cv2.IMREAD_GRAYSCALE)[np.newaxis, ...]
        return img_arr
    