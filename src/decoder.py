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

    # TODO: Create a pooler to make encoder outputs able to be inputed into decoder
    # The first layer of the decoder are embedding layers
    # If the input = (n, m), then the output of Embedding(v, h) is (n, m, h)
    # Basically we need to mimic this
    """
        I copied this from 
        VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-224-in21k", "distilgpt2")
        after comparing the encoder to
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        and assessing the differences in their heads
        (pooler): ViTPooler(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (activation): Tanh()
            )
    """
    self.hidden_dim = hidden_dim
    # self.model.transformer.wte
    # self.model.transformer.wpe

    # Freeze all other layers
    # The reasoning is that an image-captioner translates images into language space
    # Since the pre-trained decoder can already properly dechiper a latent language vector,
    # there is no need to modify it
    # for param in self.model.parameters():
    #    param.requires_grad = False
    

def forward(self, X):
    # X will be our hidden vector from the ViT encoder
    beam_output = self.model.generate(X, max_length=50, num_beams=5, early_stopping=True)
    out_seq = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
    return out_seq