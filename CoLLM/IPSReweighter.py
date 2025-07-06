import numpy as np
from scipy.sparse import coo_matrix

# Credits: https://github.com/XuChen0427/FairDiverse
# https://github.com/JiangM-C/IFairLRS/blob/main/code/finetune_gen_weight_popularity.py

def Build_Adjecent_Matrix(iid2pid_dict, item_num,group_num):
    """
        Builds an adjacency matrix based on the group-item mapping, initializing it with ones,
        and adjusting rows with no connections.

        This function uses the `Init_Group_AdjcentMatrix` to retrieve a mapping of item IDs (iid) to product IDs (pid),
        constructs an adjacency matrix, and ensures that rows with no connections are assigned a default value.
        
        :param iid2pid_dict: A dictionary mapping item IDs (iid:int) to popularity (group) IDs (pid:int).
               item_num: #items
               group_num: #groups
               
        :return: A tuple containing:
            - A 2D NumPy array representing the adjacency matrix.
            - An updated dictionary mapping popularity IDs (iid) to product IDs (pid).
    """

    iid2pid = iid2pid_dict.copy()
    row = list(iid2pid.keys())
    col = list(iid2pid.values())
    data = np.ones_like(row)
    M = coo_matrix((data, (row, col)), shape=(item_num,group_num))
    M = M.toarray()

    for i in range(len(M)):
        # Rows with no connections (not at all belonging to any item group): have default marking to first group (most popular item group)
        if np.sum(M[i]) == 0:
            M[i][0] = 1
            iid2pid[i] = 0

    return M,iid2pid


class IPS(object):
    
    def __init__(self, dataset, group_num, group_weight, variance_control):
        
        self.dataset = dataset
        item_num = 1 + len(self.dataset.item2id)  # Pass 1 + #items since iid2pid_dict starts from 1 to #items + 1 ; I will not change item number to be reindexed from 0 since that would complicate issues
        # # iid2pid_dict is int:int
        self.M, self.iid2pid = Build_Adjecent_Matrix(self.dataset.iid2pid_dict, item_num, group_num)
        self.variance_control = variance_control
        self.group_num = group_num
        self.group_weight = group_weight
        
        self.calculate_weights()
        print("Weight map: ",self.weight_map)
        
    def reset_parameters(self, **kwargs):
        self.exposure_count = np.zeros(self.group_num)
    
    def calculate_weights(self):
        
        genre_set = list(range(self.group_num))
        history_count = {_:0 for _ in genre_set}
        next_count = {_:0 for _ in genre_set}
        self.weight_map = {_:0 for _ in genre_set}
        for index in range(len(self.dataset)):
            sample = self.dataset[index]
            history_items = sample['InteractedItemIDs']
            for item in history_items:
                pop_idx = self.dataset.iid2pid_dict[int(item)]
                history_count[pop_idx] += 1
            
            pop_idx_target = self.dataset.iid2pid_dict[int(sample['TargetItemID'])]
            next_count[pop_idx_target] += 1
       
        assert len(history_count.keys()) == len(next_count.keys())
        assert len(history_count.keys()) == self.group_num
        for key in genre_set:
            self.weight_map[key] = (history_count[key] / np.sum(list(history_count.values()))) / (next_count[key] / np.sum(list(next_count.values())))
        
        return
                
    
    def reweight(self, input_dict):
        '''
            Param: input_dict: needs this one key: target_items. These would be used for the calculation as per Paper (Eq. 8, 9 and 10)
        '''
        return [self.weight_map[self.dataset.iid2pid_dict[int(item)]] for item in input_dict['target_items']]
        

    def reweight_old(self, input_dict):
        """
            Recalculates the batch weights based on the loss and the group adjacency matrix.

            This function computes new weights for each group in the batch based on the loss associated with each item and
            the group adjacency matrix. The batch weights are recalculated by considering the exposure counts

            :param Any:
            :return: A normalized vector of batch weights for each group in the batch.
        """
        
        items = input_dict['target_items']
        
        adj_matrix = self.M[items]

        B_t = np.sum(adj_matrix, axis=0, keepdims=False)
        self.exposure_count = self.exposure_count + B_t
        norm_count = self.group_weight * self.exposure_count / np.sum(self.exposure_count)
        batch_weight = np.matmul(adj_matrix, norm_count)
        batch_weight = batch_weight / np.sum(batch_weight)

        return 1/(batch_weight+self.variance_control)