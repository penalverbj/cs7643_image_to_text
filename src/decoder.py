import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Decoder(nn.Module):
    """
    DistilGPT2 language decoder
    Please see https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/gpt2#overview 
    https://huggingface.co/transformers/v3.3.1/pretrained_models.html
    https://huggingface.co/blog/how-to-generate
    https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
    for details

    Using GPT2 since it is a language decoder. DistilGPT2 has around 82M parameters,
    making it comparable to the alternative we were considering (DistilBERT)

    See below for why GPT2 over BERT in our use case
    https://www.kaggle.com/code/residentmario/notes-on-gpt-2-and-bert-models
    """

    def __init__(self, hidden_dim=768, max_outseq_len=50, num_beams=5):
        super(Decoder, self).__init__()

        # Initializing a pre-trained decoder model with a language head
        # language head = Takes embedded vectors and turns them back into language tokens
        self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

        self.hidden_dim = hidden_dim
        self.max_outseq_len = max_outseq_len
        self.num_beams = num_beams

        """
        Freeze all layers
        - The reasoning is that since the pre-trained decoder can already 
        dechiper a latent language vector, the main training to be done
        focused on the encoder. This hopefully reduces training time

        - Could potentially unfreeze later during training for fine-tuned results
        """
        for param in self.model.parameters():
            param.requires_grad = False
        

    def forward(self, X):
        """
        X will be our embedding hidden vector from the ViT encoder
        inputs_embeds=X allows us to skip the embedding layers at the start of the GPT2 model
        See https://discuss.huggingface.co/t/how-can-i-skip-gpt2lmheadmodel-embedding-layers/31648

        Additional args in generate() are used to perform beam search on space of possible sentences
        See https://huggingface.co/blog/how-to-generate

        old that didn't work
        beam_output = self.model.generate(inputs_embeds=X,
                                max_length=self.max_outseq_len,
                                num_beams=self.num_beams,
                                early_stopping=True,
                                pad_token_id=self.tokenizer.eos_token_id)
        """
        output = self.model(inputs_embeds=X)
        return output.logits
        # out_seq = self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
        # Return language tokens
        # return beam_output[0]


    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    model = Decoder()
    print(f"{model=}")
