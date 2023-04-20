import torch
import torch.nn as nn
import git
import sys

class Encoder(nn.Module):
    """
    ViT encoder based on TinyViT pre-trained model
    Please see https://github.com/microsoft/Cream/tree/main/TinyViT for details
    """
    def __init__(self, hidden_dim=768):
        super(Encoder, self).__init__()
        home_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
        sys.path.append(home_dir + '/models/TinyViT')
        self.model = torch.load(home_dir + '/models/TinyViT/tinyvit_21M.pt')

        # Freeze patch_embed and layer[0] (since they extract high-level feature representations)
        for param in self.model.patch_embed.parameters():
            param.requires_grad = False
        
        for param in self.model.layers[0].parameters():
            param.requires_grad = False

        # Remove head of model and replace with custom FCL with output hidden_dim
        # https://stackoverflow.com/questions/69376651/how-to-delete-replace-layer-in-existing-model
        # CAUTION: This is a naive implementation and we may need to replace more than just the head
        self.hidden_dim = hidden_dim
        self.model.head = nn.Linear(in_features=576, out_features=self.hidden_dim, bias=True)

    def forward(self, X):
        out = self.model(X)
        return out


    def unfreeze(self):
        for param in self.model.patch_embed.parameters():
            param.requires_grad = True
        
        for param in self.model.layers[0].parameters():
            param.requires_grad = True



if __name__ == "__main__":
    model = Encoder()
    print(f"{model=}")
