import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

# TODO: this class need to be refactored
class CustomRandomLinkSplit(RandomLinkSplit):
    """
    this customization is needed to guarantee that the train, val and test sets are not empty (useful for low density graphs)
    """
    def __call__(self, data):
        loops = 0
        
        train_set, val_set, test_set = super().__call__(data)

        if train_set.edge_label_index.size(1) == 0 or test_set.edge_label_index.size(1) == 0:
            print("empty train or test set, correcting...")
            randomIndex = 0
                
            if test_set.edge_label_index.size(1) == 0:
                # if the test set is empty, we take the first edge of the train set and put it in the test set
                # select a random edge index from the train set (from 0 to train_set.edge_label_index.size(1)) 
                randomIndex = torch.randint(0, train_set.edge_label_index.size(1), (1,)).item()
                test_set.edge_label_index = train_set.edge_label_index[:,randomIndex].unsqueeze(1)
                test_set.edge_label = train_set.edge_label[randomIndex].unsqueeze(0)
                
                # insert a negative edge in the test set
                neg_edge_index = negative_sampling(edge_index=train_set.edge_index, num_nodes=train_set.num_nodes, num_neg_samples=1)
                test_set.edge_label_index = torch.cat(
                    [test_set.edge_label_index, neg_edge_index],
                    dim=-1,
                )
                test_set.edge_label = torch.cat(
                    [test_set.edge_label, torch.zeros(1, dtype=torch.long).to(device=train_set.edge_label.device)],
                    dim=0,
                )
                
                # remove the edge from the train set
                train_set.edge_label_index = torch.cat(
                    [train_set.edge_label_index[:,:randomIndex], train_set.edge_label_index[:,randomIndex+1:]],
                    dim=-1,
                )
        return train_set, test_set
