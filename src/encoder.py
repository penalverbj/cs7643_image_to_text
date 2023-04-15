import torch
import torch.nn as nn
import git

class Encoder(nn.Module):
    """
    ViT encoder based on TinyViT pre-trained model
    Please see https://github.com/microsoft/Cream/tree/main/TinyViT for details
    """

def __init__(self, hidden_dim=128):
    super(Encoder, self).__init__()
    home_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    self.model = torch.load(home_dir + '/models/TinyViT/tinyvit_21M.pt')