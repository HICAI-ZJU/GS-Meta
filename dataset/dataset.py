import os
import os.path as osp
import re
import pickle
import random
import tqdm

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

import numpy as np

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

from .mol_features import allowable_features


class FewshotMolDataset(InMemoryDataset):
    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx, train_tasks, test_tasks]
    names = {
        'pcba': ['PCBA', 'pcba', 'pcba', -1, slice(0, 128), 118, 10],
        'muv': ['MUV', 'muv', 'muv', -1, slice(0, 17), 12, 5],
        'tox21': ['Tox21', 'tox21', 'tox21', -1, slice(0, 12), 9, 3],
        'sider': ['SIDER', 'sider', 'sider', 0, slice(1, 28), 21, 6],

        # toxcast subtask
        'toxcast-APR': ['ToxCast-APR', 'toxcast-APR', 'toxcast-APR', 0, slice(1, 44), 33, 10],
        'toxcast-ATG': ['ToxCast-ATG', 'toxcast-ATG', 'toxcast-ATG', 0, slice(1, 147), 106, 40],
        'toxcast-BSK': ['ToxCast-BSK', 'toxcast-BSK', 'toxcast-BSK', 0, slice(1, 116), 84, 31],
        'toxcast-CEETOX': ['ToxCast-CEETOX', 'toxcast-CEETOX', 'toxcast-CEETOX', 0, slice(1, 15), 10, 4],
        'toxcast-CLD': ['ToxCast-CLD', 'toxcast-CLD', 'toxcast-CLD', 0, slice(1, 20), 14, 5],
        'toxcast-NVS': ['ToxCast-NVS', 'toxcast-NVS', 'toxcast-NVS', 0, slice(1, 140), 100, 39],
        'toxcast-OT': ['ToxCast-OT', 'toxcast-OT', 'toxcast-OT', 0, slice(1, 16), 11, 4],
        'toxcast-TOX21': ['ToxCast-TOX21', 'toxcast-TOX21', 'toxcast-TOX21', 0, slice(1, 101), 80, 20],
        'toxcast-Tanguay': ['ToxCast-Tanguay', 'toxcast-Tanguay', 'toxcast-Tanguay', 0, slice(1, 19), 13, 5],
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):

        if Chem is None:
            raise ImportError('`MoleculeNet` requires `rdkit`.')

        self.name = name
        assert self.name in self.names.keys()
        super(FewshotMolDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.n_task_train, self.n_task_test = self.names[self.name][5], self.names[self.name][6]
        self.total_tasks = self.n_task_train + self.n_task_test
        if name != 'pcba':
            self.train_task_range = list(range(self.n_task_train))
            self.test_task_range = list(range(self.n_task_train, self.n_task_train + self.n_task_test))
        else:
            self.train_task_range = list(range(5, self.total_tasks - 5))
            self.test_task_range = list(range(5)) + list(range(self.total_tasks - 5, self.total_tasks))

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.index_list = pickle.load(open(self.processed_paths[1], 'rb'))
        self.y_matrix = np.load(open(self.processed_paths[2], 'rb'))

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt', 'index_list.pt', 'label_matrix.npz'

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        y_list = []
        data_id = 0
        for line in tqdm.tqdm(dataset):
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')
            smiles = line[self.names[self.name][3]]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            Chem.Kekulize(mol)

            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)
            y_list.append(ys)

            xs = []
            for atom in mol.GetAtoms():
                x = []
                x.append(allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()))
                x.append(allowable_features['possible_chirality_list'].index(atom.GetChiralTag()))
                xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, 2)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e.append(allowable_features['possible_bonds'].index(bond.GetBondType()))
                e.append(allowable_features['possible_bond_dirs'].index(bond.GetBondDir()))

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 2)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles, id=data_id)
            data_id += 1

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        y_matrix = np.array(y_list)
        index_list = []  # [[[],[]], [[],[]]], task-label-index
        for task_i in range(y_matrix.shape[1]):
            task_i_label_values = y_matrix[:, task_i]
            class1_index = np.nonzero(task_i_label_values > 0.5)[0].tolist()
            class0_index = np.nonzero(task_i_label_values < 0.5)[0].tolist()
            index_list.append([class0_index, class1_index])

        torch.save(self.collate(data_list), self.processed_paths[0])
        pickle.dump(index_list, open(self.processed_paths[1], 'wb'))
        np.save(open(self.processed_paths[2], 'wb'), y_matrix)

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))

