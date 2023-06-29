import numpy as np


def sample_from_candi(candi_list, num):
    return np.random.choice(candi_list, num, replace=False).tolist()



def dataset_sampler(dataset, n_support, n_query, tgt_id, inductive=False):
    tgt_index_list = dataset.index_list[tgt_id]
    class0_num, class1_num = len(tgt_index_list[0]), len(tgt_index_list[1])
    if class0_num > n_support and class1_num > n_support:
        support_list_i_0 = sample_from_candi(tgt_index_list[0], n_support)
        support_list_i_1 = sample_from_candi(tgt_index_list[1], n_support)
    elif class0_num <= n_support < class1_num:
        # class 0 not enough
        support_list_i_0 = sample_from_candi(tgt_index_list[0], class0_num - 1)
        support_list_i_1 = sample_from_candi(tgt_index_list[1], 2 * n_support - class0_num + 1)
    else:
        support_list_i_0 = sample_from_candi(tgt_index_list[0], 2 * n_support - class1_num + 1)
        support_list_i_1 = sample_from_candi(tgt_index_list[1], class1_num - 1)
    support_list = support_list_i_0 + support_list_i_1

    if not inductive:
        query_candi_i_0 = [idx for idx in tgt_index_list[0] if idx not in support_list]
        query_candi_i_1 = [idx for idx in tgt_index_list[1] if idx not in support_list]
        query_list = sample_from_candi(query_candi_i_0, 1) + sample_from_candi(query_candi_i_1, 1)
        query_candi = [idx for idx in query_candi_i_0 + query_candi_i_1 if idx not in query_list]
        query_list += sample_from_candi(query_candi, n_query - 2)
    else:
        query_list = [idx for idx in tgt_index_list[0] + tgt_index_list[1]
                      if idx not in support_list]
    return dataset[support_list], dataset[query_list]
