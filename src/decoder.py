import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Decoder(nn.Module):
    """
    DistilGPT2 language decoder
    Please see https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/gpt2#overview 
    https://huggingface.co/transformers/v3.3.1/pretrained_models.html
    https://huggingface.co/blog/how-to-generate for details

    Using GPT2 since it is a language decoder. DistilGPT2 has around 82M parameters,
    making it comparable to the alternative we were considering (DistilBERT)

    See below for why GPT2 over BERT in our use case
    https://www.kaggle.com/code/residentmario/notes-on-gpt-2-and-bert-models
    """

def __init__(self, hidden_dim=1000, max_outseq_len=50):
    super(Decoder, self).__init__()
    self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    # Freeze all layers
    # The reasoning is that an image-captioner translates images into language space
    # Since the pre-trained decoder can already properly dechiper a latent language vector,
    # there is no need to modify it
    for param in self.model.parameters():
        param.requires_grad = False
    

def forward(self, X):
    # X will be our hidden vector from the ViT encoder
    # TODO: Figure out how to get 
    # X = input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='tf')
    # So X is 1xn vector
    # model.generate() will take this input, but say that it was expecting int instead of floats
    # This is likely due to the embedding layer in GPT2
    # So should I just remove these layers?
    # However VisionEncoderDecoderModel.from_encoder_decoder_pretrained("microsoft/swin-base-patch4-window7-224-in22k", "distilgpt2")
    # shows that the embedding layer is still in the decoder... what should we do?
    beam_output = self.model.generate(X, max_length=50, num_beams=5, early_stopping=True)
    out_seq = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
    return out_seq