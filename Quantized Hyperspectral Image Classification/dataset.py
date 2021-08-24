import os
import h5py
import scipy.io as sio
import numpy as np
import random
import torch
import torch.utils.data as udata

def data_add_zero(data, patch_size=9):
    assert data.ndim == 3
    dx = patch_size // 2
    data_add_zeros = np.zeros((data.shape[0]+2*dx, data.shape[1]+2*dx, data.shape[2]))
    data_add_zeros[dx:-dx, dx:-dx, :] = data
    return data_add_zeros

def get_mean_data(data, patch_size=9):
    dx = patch_size // 2
    data_add_zeros = data_add_zero(data= data, patch_size=patch_size)
    for i in range(dx):
        data_add_zeros[:, i, :] = data_add_zeros[:, 2 * dx -i,:]
        data_add_zeros[i,:,:]=data_add_zeros[2*dx-i,:,:]
        data_add_zeros[:,-i-1,:]=data_add_zeros[:,-(2*dx-i)-1,:]
        data_add_zeros[-i-1,:,:]=data_add_zeros[-(2*dx-i)-1,:,:]
    data_mean = np.zeros(data.shape)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            x_start, x_end = x, x+patch_size
            y_start, y_end = y, y+patch_size
            patch = np.array(data_add_zeros[x_start:x_end, y_start:y_end, :])
            data_mean[x, y, :] = np.mean(patch.reshape(patch_size**2, patch.shape[2]), axis=0)

    return data_mean

def normalize(data):
    min_val = min(data.flatten())
    if min_val>=0:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j, :] /= np.max(np.abs(data[i,j,:]))
                data[i,j,:] = data[i,j,:] * 2 -1
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i,j,:] /= np.max(np.abs(data[i,j,:]))

    return data

def get_labels_pos(label_array, label_unique):
    label_pos = dict()
    for label in label_unique:
        label_pos[label]=[]
    for x in range(label_array.shape[0]):
        for y in range(label_array.shape[1]):
            curr_label=label_array[x,y]
            if curr_label !=0:
                label_pos[curr_label].append([x,y])

    return label_pos

def divide_labels_by_nums(label_pos, train_nums_each_class):
    label_unique = label_pos.keys()
    training_num=dict()
    for label in label_unique:
        training_num[label]=int(train_nums_each_class)
    train_label_pos=dict()
    test_label_pos=dict()

    for curr_label in label_unique:
        curr_label_pos=np.random.permutation(label_pos[curr_label])
        train_label_pos[curr_label]=curr_label_pos[: (training_num[curr_label])]
        test_label_pos[curr_label]=curr_label_pos[(training_num[curr_label]) :]
    return  train_label_pos, test_label_pos

def get_data_divided(train_label_pos, test_label_pos, data, name):
    train_nums = sum(len(train_label_pos[key]) for key in train_label_pos.keys())
    train_label, train_index = np.zeros(train_nums), np.zeros([train_nums,2], dtype='int')
    train_data = np.zeros([train_nums, data.shape[2]])
    h5f = h5py.File(name+'_train.h5', 'w')

    i = 0
    train_label_unique = train_label_pos.keys()
    for label in train_label_unique:
        curr_label_pos_array = train_label_pos[label]
        for curr_label_pos in range(len(curr_label_pos_array)):
            train_label[i] = label
            train_index[i] = curr_label_pos_array[curr_label_pos]
            train_data[i,:] = data[train_index[i][0], train_index[i][1], :]
            concat_data = np.concatenate((train_data[i, :][np.newaxis,:], np.full(train_data[i, :].shape, train_label[i])[np.newaxis,:]) , axis=0)
            h5f.create_dataset(str(train_index[i]), data=concat_data)
            i += 1

    h5f.close()
    print('\ndataset{} have {} train samples\n'.format(name, i))
    
    h5f = h5py.File(name+'_test.h5', 'w')
    test_nums = sum(len(test_label_pos[key]) for key in test_label_pos.keys())
    test_label, test_index = np.zeros(test_nums), np.zeros([test_nums,2], dtype='int')
    test_data=np.zeros([test_nums,data.shape[2]])

    i = 0
    print(sorted(test_label_pos.keys()))
    test_label_unique = test_label_pos.keys()
    for label in test_label_unique:
        curr_label_pos_array = test_label_pos[label]
        for curr_label_pos in range(len(curr_label_pos_array)):
            test_label[i] = label
            test_index[i] = curr_label_pos_array[curr_label_pos]
            test_data[i, :] = data[test_index[i][0], test_index[i][1], :]
            concat_data = np.concatenate((test_data[i, :][np.newaxis,:], np.full(test_data[i, :].shape, test_label[i])[np.newaxis,:]) , axis=0)
            h5f.create_dataset(str(test_index[i]), data=concat_data)
            i += 1

    h5f.close()
    print('\ndataset{} have {} test samples\n'.format(name, i))
    # return (train_label, train_index, train_data), (test_label, test_index, test_data)


def generate_data(pavia, pavia_label, salinas, salinas_label):
    pavia_label_unique = list(range(1,10))
    salinas_label_unique = list(range(1,17))
    pavia_label_pos=get_labels_pos(pavia_label, pavia_label_unique)
    salinas_label_pos=get_labels_pos(salinas_label, salinas_label_unique)
    pavia_train_label_pos, pavia_test_label_pos = divide_labels_by_nums(pavia_label_pos, 200)
    salinas_train_label_pos, salinas_test_label_pos = divide_labels_by_nums(salinas_label_pos, 200)
    get_data_divided(pavia_train_label_pos, pavia_test_label_pos, pavia, 'pavia')
    get_data_divided(salinas_train_label_pos, salinas_test_label_pos, salinas, 'salinas')
    
class HSIDataset(udata.Dataset):
    def __init__(self, dataset_name='pavia', mode='train'):
        self.mode = mode
        self.dataset_name = dataset_name
        if self.mode == 'train':
            h5f = h5py.File(self.dataset_name+'_train.h5', 'r')
        if self.mode == 'test':
            h5f = h5py.File(self.dataset_name+'_test.h5', 'r')
        self.keys = list(h5f.keys())
        if self.mode == 'train':
            random.shuffle(self.keys)
        else:
            self.keys.sort()
        h5f.close()
     
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            h5f = h5py.File(self.dataset_name+'_train.h5', 'r')
        if self.mode == 'test':
            h5f = h5py.File(self.dataset_name+'_test.h5', 'r')
        key = self.keys[index]
        concat_data = h5f[key]
        data = concat_data[0,:]
        label = concat_data[1,:][0]
        # label = np.array([label])
        data = torch.Tensor(data)
        label = torch.tensor(label) - 1
        h5f.close()
        return data, label
        
if __name__ == '__main__':
    path = 'G:\\bishe\\HSI'
    pavia = sio.loadmat('G:\\bishe\\HSI\\Pavia.mat')['pavia']
    pavia_gt = sio.loadmat('G:\\bishe\\HSI\\Pavia_gt.mat')['pavia_gt']
    salinas = sio.loadmat('G:\\bishe\\HSI\\Salinas_corrected.mat')['salinas_corrected']
    salinas_gt = sio.loadmat('G:\\bishe\\HSI\\Salinas_gt.mat')['salinas_gt']
    pavia_9x9_mean = get_mean_data(data=pavia, patch_size=9)
    pavia_9x9_mean = normalize(pavia_9x9_mean)
    salinas_9x9_mean = get_mean_data(data=salinas, patch_size=9)
    salinas_9x9_mean = normalize(salinas_9x9_mean)
    generate_data(pavia_9x9_mean, pavia_gt, salinas_9x9_mean, salinas_gt)



