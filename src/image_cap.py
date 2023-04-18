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
        # TODO: Not sure if this works... haven't checked if encoder output shape matches what's expected by decoder
        out = self.encoder_model(X)
        out = self.decoder_model(out)
        return out