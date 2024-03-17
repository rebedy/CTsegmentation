import os
from os.path import join
from glob import glob
import numpy as np

import torch
from torch.utils.data import Dataset

from vti_processing import vtkReader_N_2Array, vtkReader2Array_N_Spacing

# ------------------------------------------------------- #

LOAD_MODE = "ORI"
SLICE_SHUFFLE = True
SEED = 42
# ### TODO | DY : How many slices do you want as an input channel?
NUM_SLICES = 5


class PatientDataset(Dataset):  # CustomDataset
    def __init__(self, dataset_dir):
        self.data_dir = dataset_dir

    def __len__(self):  # return count of sample we have
        """ Trun over the size/length of dataset, the total number of samples."""
        return len(os.listdir(self.data_dir))

    def __getitem__(self, index):
        return self.data_dir[index]


class CTSegPathDataset(Dataset):  # CustomDataset
    def __init__(self, dataset_dir=os.getcwd()):
        """
        Where the initial logic happens like transform.
        Preprocessing dataset happens here.
        Data downloading, reading, etc.
        """
        self.data_dir = dataset_dir

    def __len__(self):  # return count of sample we have
        """ Trun over the size/length of dataset, the total number of samples."""
        return len(os.listdir(self.data_dir))

    def __getitem__(self, index):
        """ Sampling certain 1 data from dataset. Return one item on the index """

        if LOAD_MODE == 'NORM':
            ct_path_list = glob(self.data_dir + "//*//ct_norm.vti")[:20]
        else:
            ct_path_list = glob(self.data_dir + "//*//ct.vti")[:20]
        mask_path_list = glob(self.data_dir + "/*/mask.vti")[:20]

        return {'image_path': ct_path_list[index], 'mask_path': mask_path_list[index]}


class CTSeg_BatchSliceDataset(Dataset):  # CustomDataset

    def __init__(self, patient_dir):
        """
        Where the initial logic happens like transform.
        Preprocessing dataset happens here.
        Data downloading, reading, etc.
        """
        if LOAD_MODE == 'NORM':  # From CTSegPathDataset()
            ct_path = join(patient_dir, "ct_norm.vti")
            if not os.path.isfile(ct_path):
                raise FileNotFoundError
            else:
                pass
        else:
            ct_path = join(patient_dir, "ct.vti")
            if not os.path.isfile(ct_path):
                raise FileNotFoundError
            else:
                pass
        mask_path = join(patient_dir, "mask.vti")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError
        else:
            pass

        self.image_arr = vtkReader_N_2Array(ct_path)  # (1,5,H,W)
        self.image = torch.from_numpy(self.image_arr).float()
        mask_arr = vtkReader_N_2Array(mask_path)  # (1,H,W)
        self.mask = torch.from_numpy(mask_arr).long()

    def __len__(self):  # return count of sample we have
        """ Trun over the size/length of dataset, the total number of samples."""
        return self.image.shape[0] - 4

    def __getitem__(self, index):
        indices = list(range(2, len(self.image) - 2))
        if SLICE_SHUFFLE:
            np.random.seed(SEED)
            np.random.shuffle(indices)
        idx = indices[index]

        pre_idx = int(np.floor(NUM_SLICES / 2))  # == 5/2
        post_idx = NUM_SLICES - pre_idx  # == 5-2

        image_slices = self.image[idx - pre_idx: idx + post_idx]
        mask_slices = self.mask[idx]

        sample = {'image': image_slices, 'mask': mask_slices}

        return sample


class CTSeg_ValDataset(Dataset):  # CustomDataset
    def __init__(self, patient_dir=os.getcwd()):
        """
        Where the initial logic happens like transform.
        Preprocessing dataset happens here.
        Data downloading, reading, etc.
        ! Same as CTSeg_BatchSliceDataset except this returns vti spacing values.
        """
        # ### TODO | DY : This is for the sake of when there is more than one Patient Batch.
        if LOAD_MODE == 'NORM':  # From CTSegPathDataset()
            ct_path = join(patient_dir + "/ct_norm.vti")
            if not os.path.isfile(ct_path):
                print("If args.source_company is 'YYY', check the LOAD_MODE.")
                raise FileNotFoundError
            else:
                pass
        else:
            ct_path = join(patient_dir + "/ct.vti")
            if not os.path.isfile(ct_path):
                raise FileNotFoundError
            else:
                pass
        mask_path = join(patient_dir + "/mask.vti")
        if not os.path.isfile(mask_path):
            print("If SOURCE_COMPANY is 'YYY', check the patient number.")
            raise FileNotFoundError
        else:
            pass
        image_arr, self.image_sp = vtkReader2Array_N_Spacing(ct_path)
        self.image = torch.from_numpy(image_arr).float()
        mask_arr = vtkReader_N_2Array(mask_path)
        self.mask = torch.from_numpy(mask_arr).long()  # ! GT have to be Long type

    def __len__(self):  # return count of sample we have
        """ Trun over the size/length of dataset, the total number of samples."""
        return self.image.shape[0] - 4

    def __getitem__(self, index):

        indices = list(range(2, len(self.image) - 2))
        idx = indices[index]

        pre_idx = int(np.floor(NUM_SLICES / 2))  # == 5/2
        post_idx = NUM_SLICES - pre_idx  # == 5-2

        image_slices = self.image[idx - pre_idx: idx + post_idx]
        mask_slices = self.mask[idx]

        sample = {'image': image_slices, 'mask': mask_slices, 'image_sp': self.image_sp}

        return sample


class CTSegCustomDataset(Dataset):  # CustomDataset
    def __init__(self, patient_dir=os.getcwd()):
        """
        Where the initial logic happens like transform.
        Preprocessing dataset happens here.
        Data downloading, reading, etc.
        ! Same as CTSeg_BatchSliceDataset except this returns vti spacing values.
        """
        if not os.path.isfile(patient_dir):
            raise FileNotFoundError
        else:
            pass
        image_arr, self.image_sp = vtkReader2Array_N_Spacing(patient_dir)
        self.image = torch.from_numpy(image_arr).float()

    def __len__(self):  # return count of sample we have
        """ Trun over the size/length of dataset, the total number of samples."""
        return self.image.shape[0] - 4

    def __getitem__(self, index):

        indices = list(range(2, len(self.image) - 2))
        idx = indices[index]

        pre_idx = int(np.floor(NUM_SLICES / 2))  # == 5/2
        post_idx = NUM_SLICES - pre_idx  # == 5-2

        image_slices = self.image[idx - pre_idx: idx + post_idx]

        sample = {'image': image_slices, 'image_sp': self.image_sp}

        return sample


def CTSegSampler():
    def __int__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.dataset))

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
