from argparse import ArgumentParser

import torch
import torch.optim as optim
from ignite.metrics import Rouge

from tqdm import tqdm

from image_cap import ImageCap
import git
from glob import glob
import numpy as np

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1.0e-3


def distill(hidden_dim: int = 768, max_outseq_len: int = 50, num_beams: int = 5,
            tl_alpha: int = 1, tl_beta: int = 1, tl_gamma: int = 0):
    """
    This will call our image_cap.py and the forward-pass outputs of teacher.py to perform knowledge distillation
    According to TinyViT, one way to make this less resource intensive is to 
    perform inference with our teacher prior to training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImageCap(hidden_dim=hidden_dim, max_outseq_len=max_outseq_len, num_beams=num_beams).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-4)
    optimizer.zero_grad()
    accuracyMetric = Rouge(variants=["L"], multiref="average")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Load teacher outputs
    home_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
    csv_path = glob(home_dir + "/data/teacherResults.csv")[0]
    teacher_labels = np.genfromtxt(csv_path, delimiter=',')

    # self.tl_alpha = tl_alpha
    # self.tl_beta = tl_beta
    # self.tl_gamma = tl_gamma

    # TODO: Create teacher embeddings
    # TODO: Question: what kind of loss even makes sense? Is cosine similar really the best thing?
    # TODO: On that note, what output should be passed to the loss? Embeddings, categorical dist, tokens?
    # Resources:
    # https://alexnim.com/coding-projects-knowledge-distillation.html
    # L = alpha * L_distill + beta * L_training + gamma * L_cosine

    # https://blog.floydhub.com/knowledge-distillation/
    #   (1 - alpha) * cross_entropy_loss wrt hard labels +  alpha * KL-Div wrt soft teacher labels
    # https://arxiv.org/pdf/1910.01108.pdf Section 2

    def ideal_label(self, id_):
        # since we have 5 candidate teacher embeddings, a few options here.
        # we could try to minimize against the closest embedding
        # we could try to minimize against the furthest embedding
        # we could use a mix, sum, etc.
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

    for epoch in range(EPOCHS):
        print(f"==============================================")
        print(f"EPOCH: {epoch}")
        print(f"==============================================")
        epoch_loss = 0

        for batch_id, data in enumerate(tqdm(train_data_loader)):
            batch_loss = 0
            # TODO: Implement forward pass, calculate loss, and backward pass
            out = self.model.forward()
            for image in out:
                batch_loss += triple_loss(out)
            batch_loss.backward()
            epoch_loss += batch_loss

            print("success")


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
