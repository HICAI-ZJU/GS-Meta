from copy import deepcopy
import torch

nan_type_dict = {'0': 1, '1': 2, 'nan': 3}


def get_edge(label_matrix, label, shift, bi_direct=True):
    if label is None:
        # Each row in the result contains the indices of a non-zero element in input.
        # sample -> task
        edge = torch.nonzero(torch.isnan(label_matrix), as_tuple=False)
    else:
        edge = torch.nonzero(label_matrix == label, as_tuple=False)
    edge[:, 0] = edge[:, 0] + shift
    if bi_direct:
        edge_copy = deepcopy(edge)
        edge_copy[:, 0], edge_copy[:, 1] = edge[:, 1], edge[:, 0]
        edge = torch.cat([edge, edge_copy])  # [n_edge, 2]
    return edge


class InductiveGraphConnector:
    @staticmethod
    def connect_task_and_sample(support_y, query_y, nan_w=0., nan_type='nan'):
        """
        :param support_y: [n_s, y]
        :param query_y: [n_q, y]
        :param nan_w: int
        :param nan_type
        :return: tgt_s_y: [n_s, 1]
                 tgt_q_y: [n_q, 1]
                 tgt_s_idx: [n_s, 2]
                 tgt_q_idx: [1, 2]
                 edge_ls: List of Tensor: [2, edge]
                 edge_type_ls: List of Tensor: [n_edge, 1]
                 edge_w_ls: List of Tensor: [n_edge, 1]
        """
        nan_idx = nan_type_dict[nan_type]
        tgt_s_y, tgt_q_y = support_y[:, [0]], query_y[:, [0]]  # [n_s,1], [n_q,1]
        n_task = support_y.shape[1]
        n_q, n_s = query_y.shape[0], support_y.shape[0]
        support_y = support_y.repeat(n_q, 1, 1)  # [n_q, n_s, y]
        query_y = query_y.unsqueeze(1)  # [n_q, 1, y]
        concat_y = torch.cat([support_y, query_y], dim=1)  # [n_q, n_s+1, y]
        edge_ls, edge_type_ls, edge_w_ls = [], [], []
        for y_i in concat_y:
            # y_i: [n_s+1, y]
            support_y_i = y_i[:-1, :]  # [n_s, y]
            auxi_query_y_i = y_i[[-1], 1:]  # [1, y-1]
            support_edge_i_0 = get_edge(support_y_i, 0, n_task)
            support_edge_i_1 = get_edge(support_y_i, 1, n_task)
            query_edge_i_0 = get_edge(auxi_query_y_i, 0, n_task + n_s)
            query_edge_i_1 = get_edge(auxi_query_y_i, 1, n_task + n_s)
            if nan_w == 0:
                edge_i = torch.cat([support_edge_i_0, query_edge_i_0,
                                    support_edge_i_1, query_edge_i_1])
                edge_type_i = [1] * (len(support_edge_i_0) + len(query_edge_i_0)) + \
                              [2] * (len(support_edge_i_1) + len(query_edge_i_1))
                edge_type_i = torch.tensor(edge_type_i).to(edge_i.device)
                edge_w_i = torch.tensor([1.] * len(edge_type_i)).to(edge_i.device)
            else:
                support_edge_i_nan = get_edge(support_y_i, None, n_task)
                query_edge_i_nan = get_edge(auxi_query_y_i, None, n_task + n_s)
                edge_i = torch.cat([support_edge_i_0, query_edge_i_0,
                                    support_edge_i_1, query_edge_i_1,
                                    support_edge_i_nan, query_edge_i_nan])
                edge_0_n, edge_1_n, edge_nan_n = len(support_edge_i_0) + len(query_edge_i_0), \
                                                 len(support_edge_i_1) + len(query_edge_i_1), \
                                                 len(support_edge_i_nan) + len(query_edge_i_nan)
                edge_type_i = [1] * edge_0_n + [2] * edge_1_n + [nan_idx] * edge_nan_n
                edge_type_i = torch.tensor(edge_type_i).to(edge_i.device)
                edge_w_i = torch.tensor([1.] * (edge_0_n + edge_1_n) + [nan_w] * edge_nan_n).to(edge_i.device)
            edge_i = edge_i.transpose(0, 1)  # [2, n_edge]
            edge_w_i = edge_w_i.unsqueeze(-1)  # [n_edge, 1]
            edge_type_i = edge_type_i.unsqueeze(-1)  # [n_edge, 1]
            edge_ls.append(edge_i)
            edge_type_ls.append(edge_type_i)
            edge_w_ls.append(edge_w_i)
        tgt_s_idx = torch.tensor([list(range(n_task, n_task + n_s)),  # sample
                                  [0] * n_s]).transpose(0, 1)  # task, [n_s, 2]
        tgt_q_idx = torch.tensor([[n_task + n_s], [0]]).transpose(0, 1)  # [1, 2]
        tgt_s_idx, tgt_q_idx = tgt_s_idx.to(support_y.device), tgt_q_idx.to(support_y.device)
        return tgt_s_y, tgt_q_y, tgt_s_idx, tgt_q_idx, edge_ls, edge_type_ls, edge_w_ls

    @staticmethod
    def connect_graph(adj, edge_ls, edge_type_ls, edge_w_ls, n_task):
        """
        :param adj: [n_q, 1, n_s+1, n_s+1]
        :param edge_ls: List of Tensor: [2, n_edge]
        :param edge_type_ls: List of Tensor: [n_edge, 1]
        :param edge_w_ls: list of Tensor: [n_edge, 1]
        :param n_task:
        :return: edges: List of Tensor: [2, n_edge]
                 edge_types: List of Tensor: [n_edge, 1]
                 edge_ws: List of Tensor: [n_edge, 1]
        """
        edges, edge_types, edge_ws = [], [], []
        for i in range(len(adj)):
            adj_i = adj[i].squeeze(0)  # [n_s+1, n_s+1]
            edge_i, edge_type_i, edge_w_i = edge_ls[i], edge_type_ls[i], edge_w_ls[i]
            sample_edge = torch.nonzero(adj_i > 0, as_tuple=False)  # [n_edge, 2]
            sample_w = adj_i[sample_edge[:, 0], sample_edge[:, 1]]
            if sample_edge.shape[0] != 0:
                sample_edge = sample_edge + n_task
                edge_i = torch.cat([edge_i, sample_edge.transpose(0, 1)], dim=1)  # [2, n_edge]
                edge_type_i = torch.cat(
                    [edge_type_i, torch.tensor([0] * sample_edge.shape[0]).unsqueeze(1).to(adj.device)],
                    dim=0)  # [n_edge, 1]
                edge_w_i = torch.cat([edge_w_i, sample_w.unsqueeze(1)], dim=0)  # [n_edge, 1]
            edges.append(edge_i)
            edge_types.append(edge_type_i)
            edge_ws.append(edge_w_i)
        return edges, edge_types, edge_ws
