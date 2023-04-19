import git
import sys
import requests
from PIL import Image
# from datasets import load_dataset
from dataHelpers import cocoLoader

from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

class Teacher():
    def __init__(self, data_set="", captioning_model="nlpconnect/vit-gpt2-image-captioning"):
        # load a fine-tuned image captioning model and corresponding tokenizer and image processor
        self.model = VisionEncoderDecoderModel.from_pretrained(captioning_model)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(captioning_model)
        self.image_processor = ViTImageProcessor.from_pretrained(captioning_model)
        self.data_set = data_set

    def process_batch(self):
        pass

    def load_data(self):
        home_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
        sys.path.append(home_dir + '/models/TinyViT')
        loader = cocoLoader.cocoLoader("C:/Users\penal\DeepLearning/final\data\coco/annotations\captions_train2017.json", "data/coco/train2017")
        self.data_set = loader.get_all_imgs()
        print("DONE LOADING")
    def get_pixels_single_image(self, image):
        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # image = Image.open(requests.get(url, stream=True).raw)
        # loader = cocoLoader.cocoLoader("C:/Users\penal\DeepLearning/final\data\coco/annotations\captions_train2017.json", "data/coco/train2017")
        # imgs = loader.get_imgs(1, random=True)
        # image = imgs[0]['image']
        # print(imgs[0]['annotations'])
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        return pixel_values

    def caption_single_image(self, pixel_values):
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)


def test():
    teacher = Teacher()
    teacher.load_data()
    # pixel_values = teacher.get_pixels_single_image()
    # teacher.caption_single_image(pixel_values)


if __name__ == "__main__":
    test()

"""
This will implement the forward pass of our teacher model
See https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder for potential implementation

Also see decoder.py comments and links for more implementation details!


Below is an example of how to implement an pre-trained teacher:

import requests
from PIL import Image

from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

# load a fine-tuned image captioning model and corresponding tokenizer and image processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# let's perform inference on an image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = image_processor(image, return_tensors="pt").pixel_values

# autoregressively generate caption (uses greedy decoding by default)
generated_ids = model.generate(pixel_values)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

# Our own code for debugging
encoder = model.encoder
decoder = model.decoder
test_encoded = encoder(pixel_values)
test_encoded.last_hidden_state.shape
test_encoded.pooler_output.shape
test_decoded = decoder.generate(inputs_embeds=test_encoded.last_hidden_state)
tokenizer.batch_decode(test_decoded, skip_special_tokens=True)[0]
"""
