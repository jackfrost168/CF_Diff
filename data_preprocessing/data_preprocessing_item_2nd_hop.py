import numpy as np
import torch
import scipy.sparse as sp


def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)
    print(train_list)
    print(len(train_list))

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


data_path = './datasets/yelp2018/'

dataset_name = 'yelp2018'

train_path = data_path + 'train_list_' + dataset_name + '.npy'
valid_path = data_path + 'valid_list_' + dataset_name + '.npy'
test_path = data_path + 'test_list_' + dataset_name + '.npy'
print(train_path)
print(valid_path)
print(test_path)

train_data, valid_y_data, test_y_data, n_user, n_item = data_load(train_path, valid_path, test_path)
print(train_data.shape)
#print(train_data.nbytes)

data = train_data.todense().A
print(data.shape)
print(data.nbytes)
#valid = valid_y_data.todense().A
#test = test_y_data.todense().A
#
print("ints:", np.sum(np.sum(data, axis=1)))


def get_2hop_item_based(data):
    # Initialize an empty tensor
    sec_hop_infos = torch.empty(len(data), len(data[0]))
    print(sec_hop_infos.size())

    # Loop to add data to the tensor
    sec_hop_inters = torch.sum(data, axis=0) / n_user
    for i, row in enumerate(data):

        zero_indices = torch.nonzero(row<0.000001).t()#.squeeze()
        if i % 1000 == 0:
          print(i)

        sec_hop_infos[i] = sec_hop_inters
        sec_hop_infos[i][zero_indices[0]] = 0

    #tensor = torch.cat((data, sec_hop_infos), dim=1)  # Concatenate the data to the tensor

    return sec_hop_infos

# Call the function
hop2_rates_test = get_2hop_item_based(torch.tensor(data, dtype=torch.float32))

# Print the resulting tensor
print(hop2_rates_test.size())

# filename = "datasets/yelp2018/two_hop_rates_items_yelp2018.pt"
# torch.save(hop2_rates_test, filename)