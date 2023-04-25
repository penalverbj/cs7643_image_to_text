import torch.nn as nn
from transformers import AutoModel


class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        clip = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision = clip.vision_model
        self.fc = nn.Linear(1024, 384)

    def forward(self, x):
        out = self.vision(x)['pooler_output']
        return self.fc(out)
