import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class ImageCap(nn.Module):
    """
        Combine encoder.py and decoder.py into a fully functioning image-captioning model
    """
    def __init__(self, hidden_dim=768, max_outseq_len=50, num_beams=5):
        super(ImageCap, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_outseq_len = max_outseq_len
        self.num_beams = num_beams

        self.encoder_model = Encoder(hidden_dim=self.hidden_dim)
        self.decoder_model = Decoder(hidden_dim=self.hidden_dim,
                                     max_outseq_len=self.max_outseq_len,
                                     num_beams=self.num_beams)
        
    def forward(self, X):
        # TODO: Error. Encoder output shape doesn't match what's expected by decoder
        # encoder currently outputs (1, hidden_dim)
        # while decoder wants (1, n, hidden_dim), where n can be anything
        # This because the output of the encoder was originally a vector for classification
        # While while decoder is expecting an embeddeding for each nth word
        # How do we get the last hidden state of TinyViT to pass into decoder?
        # With huggingface output, this is easy, but idk for custom model
        # This is the equiv of getting the hidden state for each image patch
        # Do we use hooks??
        # https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
        # Wait! See line 578 of https://github.com/microsoft/Cream/blob/main/TinyViT/models/tiny_vit.py
        # We may just need to get rid of that mean!
        # encoder_model.model.forward_features(pixel_values)
        # Plan: Create another method in tiny_vit.py that is the same as foward_features, but does not
        # do the mean!
        out = self.encoder_model.forward(X)
        out = self.decoder_model.forward(out)
        return out