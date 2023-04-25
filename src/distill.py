from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
from ignite.metrics import Rouge

from tqdm import tqdm

from image_cap import ImageCap
import git
from glob import glob
import numpy as np

from pycocotools.coco import COCO
from transformers import GPT2Tokenizer

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1.0e-3


def distill(hidden_dim: int = 768, max_outseq_len: int = 25, num_beams: int = 5,
            tl_alpha: int = 1, tl_beta: int = 1, tl_gamma: int = 0):
    """
    This will call our image_cap.py and the forward-pass outputs of teacher.py to perform knowledge distillation
    According to TinyViT, one way to make this less resource intensive is to 
    perform inference with our teacher prior to training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImageCap(hidden_dim=hidden_dim, max_outseq_len=max_outseq_len, num_beams=num_beams).to(device).half()
    tokenizer = GPT2Tokenizer.from_pretrained(model.decoder_model.model_name)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-4)
    optimizer.zero_grad()
    accuracyMetric = Rouge(variants=["L"], multiref="average")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Load data
    training_data = DistillDataset(device=device, tokenizer=tokenizer,)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True,)
    # train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

    # self.tl_alpha = tl_alpha
    # self.tl_beta = tl_beta
    # self.tl_gamma = tl_gamma

    # TODO: Create teacher embeddings
    # TODO: Question: what kind of loss even makes sense? Is cosine similar really the best thing?
    # TODO: On that note, what output should be passed to the loss? Embeddings, categorical dist, tokens?
    # Resources:
    # https://alexnim.com/coding-projects-knowledge-distillation.html
    # L = alpha * L_distill + beta * L_training + gamma * L_cosine
    '''
    # https://blog.floydhub.com/knowledge-distillation/
    #   (1 - alpha) * cross_entropy_loss wrt hard labels +  alpha * KL-Div wrt soft teacher labels
    # https://arxiv.org/pdf/1910.01108.pdf Section 2

    # TODO: 
    def ideal_label(self, id_):
        # since we have 5 candidate teacher embeddings, a few options here.
        # we could try to minimize against the closest embedding
        # we could try to minimize against the furthest embedding
        # we could use a mix, sum, etc.
        # we could just pick one.
        # or we could randomize.
        pass

    def distillation_loss(self, id_):
        return tl_alpha * torch.nn.KLDivLoss(student[id_], teacher[id_])

    def training_loss(self, id_):
        return tl_beta * torch.nn.KLDivLoss(student[id_], ideal_label(id_))

    def alignment_loss(self, id_):
        if tl_gamma == 0:
            return 0
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return tl_gamma * cos(student[id_], teacher[id_])

    def triple_loss(self, id_):
        return self.distillation_loss(id_) + self.training_loss(id_) + self.alignment_loss(id_)
    '''
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()

    for epoch in range(EPOCHS):
        print(f"==============================================")
        print(f"EPOCH: {epoch}")
        print(f"==============================================")
        epoch_loss = 0

        for batch_id, data in enumerate(tqdm(train_dataloader)):
            batch_loss = 0
            # TODO: Implement forward pass, calculate loss, and backward pass
            images, teacher_embeddings, teacher_tokens, coco_tokens = data

            # target_embeddings.size = 16, 5, 1024, only with max_length padding strategy,
            # which is subject to change
            # TODO: loss function needs to be called relative to each of these targets

            # out_logits.size = (batch_size, vocab_size)
            # need the attention outputs - model.decoder_out['decoder_out_hidden'], size = (batch_size, hidden_dim)
            out_logits = model.forward(images)

            # Distillation loss???
            # What we're supposed to do: teacher_tokens = [0.1, 0.3, 0, 0.6, 0.1]
            # What we have: teacher_tokens = [0, 0, 0, 1, 0]
            
            batch_loss += ce_loss(out_logits, teacher_tokens)
            # Supervised training loss
            batch_loss += ce_loss(out_logits, coco_tokens)
            # Cosine embedding loss
            batch_loss += cosine_loss(model.decoder_out['decoder_out_hidden'], teacher_embeddings)

            batch_loss.backward()
            epoch_loss += batch_loss

            print("success")
        
        # TODO: validation testing here...


# Load teacher decoder last hidden state values into memory
# Assuming rows in teacherHidden.csv correspond 1-to-1 with teacherResults.csv
def load_teacher_data(device="cpu"):
    home_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    hidden_csv = glob(home_dir + "/data/teacher_out/teacherHidden.csv")[0]

    # Store teacher decoder last hidden state values
    hidden_values = np.genfromtxt(hidden_csv, delimiter=',')
    hidden_values = torch.from_numpy(hidden_values).to(device)

    # Next store teacher text and associations with images
    # I'm doing it this way because file names have been changed
    # and output string have ',' while being stored in csv file
    results_csv = glob(home_dir + "/data/teacher_out/teacherResults.csv")[0]
    img_id_list = []
    jpg_list = []
    annotation_list = []
    with open(results_csv, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                cut_loc = line.find(',')
                img_id = line[0:cut_loc]
                img_id_list.append(int(img_id))
                jpg_name = home_dir + '/data/coco/train2017/' + '0'*(12-len(img_id)) + img_id + '.jpg'
                annotation = line[cut_loc+1:]
                jpg_list.append(jpg_name)
                annotation_list.append(annotation)
                
            # Skip header
            except ValueError:
                pass      

    return hidden_values, img_id_list, jpg_list, annotation_list


class DistillDataset(Dataset):
    """
    Create a custom pytorch dataset class to facilitate batch training
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(
        self,
        tokenizer,
        coco_json="./data/coco/annotations/captions_train2017.json",
        device="cpu",
        img_size=(224, 224),
    ):
        print("Loading images and annotations for distillation")
        self.device = device
        self.teacher_hidden, self.img_id_list, self.jpg_list, self.teacher_ann = load_teacher_data(device=self.device)
        self.coco = COCO(coco_json)
        self.transforms = transforms.Compose([transforms.Resize(img_size)])
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return self.teacher_hidden.shape[0]

    def __getitem__(self, idx):
        # Images to pixel values
        # TODO: Check so see if concordant with expected preprocessing
        img_path = self.jpg_list[idx]
        img_id = self.img_id_list[idx]
        image = self.transforms(read_image(img_path)).to(self.device).float().half()

        teacher_embeddings = self.teacher_hidden[idx, :]

        # Some COCO images have more than 5 associated annotations
        ann_ids = self.coco.getAnnIds(img_id)
        coco_annotations = self.coco.loadAnns(ann_ids)
        if len(coco_annotations) > 5:
            coco_annotations = coco_annotations[:5]

        # Tokenize coco_annotations into vector
        # TODO: consider making padding shorter (padding='longest')
        coco_tokens = []
        for ann in coco_annotations:
            tokens = self.tokenizer(
                ann['caption'],
                return_tensors="pt",
                padding='max_length',
                truncation=True,
            )
            coco_tokens.append(tokens['input_ids'])

        coco_tokens = torch.cat(coco_tokens)

        # Tokenize teacher_embeddings
        teacher_tokens = self.tokenizer(
            teacher_embeddings['caption'],
            return_tensors="pt",
            padding='max_length',
            truncation=True,
        )
        teacher_tokens = torch.cat(teacher_tokens)

        return image, teacher_embeddings, teacher_tokens, coco_tokens
    

if __name__ == "__main__":
    parser = ArgumentParser(description="CS7643 Best Group Student Distillation script.")
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=768,
        help="Number of hidden dimensions in student."
    )
    parser.add_argument(
        '--max_outseq_len',
        type=int,
        default=50,
        help="Maximum output sentence length of student."
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=5,
        help="Number of beams in beam search when assessing most probable words outputed by student"
    )


    args = parser.parse_args()

    distill(**vars(args))
