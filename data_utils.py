import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
import torch


def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)


class DataDiffusion2(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        return item1, item2
    def __len__(self):
        return len(self.data1)


class DataDiffusion3(Dataset):
    def __init__(self, data1, data2, data3):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
    def __getitem__(self, index):
        item1 = self.data1[index]
        item2 = self.data2[index]
        item3 = self.data3[index]
        return item1, item2, item3
    def __len__(self):
        return len(self.data1)


def get_top_k_similar_pearson(data, k):
    # Subtract the mean of each row from the rows (center the data)
    mean_centered_data = data - data.mean(dim=1, keepdim=True)

    # Compute the covariance matrix
    covariance_matrix = torch.mm(mean_centered_data, mean_centered_data.t())

    # Normalize the covariance matrix to get Pearson correlation coefficients
    # Calculate the standard deviation for each row
    std_dev = mean_centered_data.norm(p=2, dim=1, keepdim=True)

    # Avoid division by zero in case there is a row with zero variance
    std_dev[std_dev == 0] = 1

    # Pearson correlation matrix
    pearson_correlation_matrix = covariance_matrix / torch.mm(std_dev, std_dev.t())

    # We need to zero out the diagonal elements (self-correlation) before getting top-k
    # Fill diagonal with very low value which cannot be a top correlation
    eye = torch.eye(pearson_correlation_matrix.size(0), device=pearson_correlation_matrix.device)
    pearson_correlation_matrix -= eye * 2  # Subtract 2 which is definitely out of bound for correlation

    # Get top-k values along each row
    _, indices = torch.topk(pearson_correlation_matrix, k=k, dim=1)

    return indices
