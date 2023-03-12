import os
import sys
path2file  = os.path.abspath(__file__)
dir2append = '/'.join(path2file.split('/')[:-2])
sys.path.append(dir2append)
print('stuff will be saved at -\n', dir2append)

import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloader import PflowDataset, collate_graphs, PflowSampler

import numpy as np
import matplotlib.pyplot as plt


config_path = sys.argv[1]
with open(config_path, 'r') as fp:
    config = json.load(fp)
# config['reduce_ds'] = -1


dataset = PflowDataset(config['path_to_train'], config, bool_inc=config['bool_inc'], reduce_ds=config['reduce_ds_train'])
loader = DataLoader(dataset, num_workers=0, batch_size=1, 
                collate_fn=collate_graphs, shuffle=False, pin_memory=False)


incidence_list = []
for _, incidence_truth, ptetaphi_truth, _, _ in tqdm(loader):

	non_garbage_mask = (ptetaphi_truth[:,:,0] != 0) * (ptetaphi_truth[:,:,1] != 0) * (ptetaphi_truth[:,:,2] != 0)
	incidence = incidence_truth[non_garbage_mask]

	incidence_list.extend(incidence.reshape(-1).tolist())


plt.hist(incidence_list, bins=100)
plt.title('incidence_dist')
path2save = os.path.join(dir2append, 'incidence_info', 'incidence_dist.png')
plt.savefig(path2save)
plt.clf()

plt.hist(incidence_list, bins=100, log=True)
plt.title('incidence_dist_log')
path2save = os.path.join(dir2append, 'incidence_info', 'incidence_dist_log.png')
plt.savefig(path2save)
plt.clf()


# negative guys
x, y, _ = plt.hist(dataset.neg_contribs, bins=100)
plt.title('negative energy contributions (MeV)')
path2save = os.path.join(dir2append, 'incidence_info', 'neg_contribs.png')
plt.savefig(path2save)
plt.clf()

np_path2save = os.path.join(dir2append, 'incidence_info', 'neg_contribs')
np.savez(np_path2save, x, y)



# fake TCs (after fixing negative guys) with zero
x, y, _ = plt.hist(dataset.fake_TC_count, bins=10, range=[0,10])
plt.title('fake TC count (after negative enrgy fix)')
path2save = os.path.join(dir2append, 'incidence_info', 'fake_tc_count.png')
plt.savefig(path2save)
plt.clf()

# fake TCs (after fixing negative guys) w/o zero
x, y, _ = plt.hist([x for x in dataset.fake_TC_count if x>0], bins=10, range=[0,10])
plt.title('fake TC count (after negative enrgy fix), no zero')
path2save = os.path.join(dir2append, 'incidence_info', 'fake_tc_count_no_zero.png')
plt.savefig(path2save)
plt.clf()




incidence_list = np.array(incidence_list)
incidence_list = incidence_list[(incidence_list != 0) * (incidence_list != 1)]
x, y, _ = plt.hist(incidence_list, bins=20)
plt.title('incidence_dist (!0 and !1)')
path2save = os.path.join(dir2append, 'incidence_info', 'incidence_dist_no_0_1.png')
plt.savefig(path2save)
plt.clf()

np_path2save = os.path.join(dir2append, 'incidence_info', 'incidence_dist_no_0_1_20bins')
np.savez(np_path2save, x, y)



incidence_list = np.array(incidence_list)
incidence_list = incidence_list[(incidence_list > 0.01) * (incidence_list < 0.99)]
y, x, _ = plt.hist(incidence_list, bins=20)
plt.title('incidence_dist (>0.01 and <0.99 )')
path2save = os.path.join(dir2append, 'incidence_info', 'incidence_dist_0.01_0.99_20bins.png')
plt.savefig(path2save)
plt.clf()

np_path2save = os.path.join(dir2append, 'incidence_info', 'incidence_dist_0.01_0.99_20bins')
np.savez(np_path2save, x, y)