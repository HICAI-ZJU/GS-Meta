from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_encoder import GNN_Encoder
from .relation import TaskRelationNet


class GSMeta(nn.Module):
    def __init__(self, task_num, train_task_num, args):
        super(GSMeta, self).__init__()
        self.mol_encoder = GNN_Encoder(num_layer=args.mol_num_layer,
                                       emb_dim=args.emb_dim,
                                       JK=args.JK,
                                       drop_ratio=args.mol_dropout,
                                       graph_pooling=args.mol_graph_pooling,
                                       gnn_type=args.mol_gnn_type,
                                       batch_norm=args.mol_batch_norm,
                                       load_path=args.mol_pretrain_load_path)

        self.relation_net = TaskRelationNet(in_dim=args.emb_dim,
                                            num_layer=args.rel_layer,
                                            edge_n_layer=args.rel_edge_n_layer,
                                            edge_hidden_dim=args.rel_edge_hidden_dim,
                                            total_tasks=task_num,
                                            train_tasks=train_task_num,
                                            batch_norm=args.rel_batch_norm,
                                            top_k=args.rel_top_k,
                                            dropout=args.rel_dropout,
                                            pre_dropout=args.rel_pre_dropout,
                                            nan_w=args.rel_nan_w,
                                            nan_type=args.rel_nan_type,
                                            edge_type=args.rel_edge_type)

    def encode_mol(self, data):
        return self.mol_encoder(data.x, data.edge_index, data.edge_attr, data.batch)

    def forward(self, s_data, q_data, s_y, q_y, sampled_task):
        s_feat, q_feat = self.encode_mol(s_data), self.encode_mol(q_data)
        s_feat_input = s_feat.repeat(q_feat.shape[0], 1, 1)  # [n_q, n_s, d]
        q_feat_input = q_feat.unsqueeze(1)  # [n_q, 1, d]
        sample_feat = torch.cat([s_feat_input, q_feat_input], dim=1)  # [n_q, n_s+1, d]
        s_logit, q_logit, s_label, q_label, graph_f = self.relation_net.forward_inductive(sample_feat, sampled_task,
                                                                                          s_y, q_y)
        return s_logit, q_logit, s_label, q_label, graph_f
