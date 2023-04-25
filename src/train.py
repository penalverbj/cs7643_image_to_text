from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from dataHelpers.diffusiondb_data_loader import get_diffusion_db_train_test_valid_dataset

from clip import CLIP
from loss import cosine_similarity_loss

from plot import plot_avg_train_valid_loss


# Save model and optimizer checkpoint flags
SAVE_OPTIM_CHECKPOINT = True
SAVE_MODEL_CHECKPOINT = True

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1.0e-3
UNFREEZE_START = 18         # set it to lower number when more samples are included.


def load_pretrained_model(device: str = 'cpu'):
    model = CLIP()

    trainable_model_weights = False
    for name, child in model.named_children():
        if name == 'vision':
            for pn, p in child.named_parameters():
                if str(UNFREEZE_START) in pn:
                    """start unfreezing layer , the weights are trainable"""
                    trainable_model_weights = True
                p.requires_grad = trainable_model_weights
                if p.requires_grad:
                    print(f"{pn} is set to be trainable.")

    return model.to(device)

def train(
    sentence_transformer_model,
    vision_captioning_model,
    data_loader,
    optimizer,
    scheduler=None,
    device='cpu',
):
    total_epoch_loss = 0
    for _, data in enumerate(tqdm(data_loader)):
        images = data['pixel_values']['pixel_values'].squeeze(1).to(device)
        prompts = data['prompt']
        target_embeddings_np = sentence_transformer_model.encode(prompts)
        target_embeddings = torch.Tensor(target_embeddings_np).to(device)

        prediction_embeddings = vision_captioning_model(images)
        loss = cosine_similarity_loss(
            prediction_embeddings=prediction_embeddings,
            target_embeddings=target_embeddings,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(vision_captioning_model.parameters(), 0.5)
        optimizer.step()

        total_epoch_loss += loss.item()
    
    if scheduler is not None:
        scheduler.step(total_epoch_loss)

    return total_epoch_loss, total_epoch_loss / len(data_loader)


def evaluate(
    sentence_transformer_model,
    vision_captioning_model,
    data_loader,
    device='cpu'
):
    total_epoch_loss = 0
    with torch.no_grad():
        for _, data in enumerate(tqdm(data_loader)):
            images = data['pixel_values']['pixel_values'].squeeze(1).to(device)
            prompts = data['prompt']
            target_embeddings_np = sentence_transformer_model.encode(prompts)
            target_embeddings = torch.Tensor(target_embeddings_np).to(device)

            prediction_embeddings = vision_captioning_model(images)
            loss = cosine_similarity_loss(prediction_embeddings, target_embeddings)
            total_epoch_loss += loss.item()

    return total_epoch_loss, total_epoch_loss / len(data_loader)


def main():
    """Main CS7643 awesomest group training loop.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # NOTE: set this db_type to something like 100k or 500k during real training
    diffusion_db = get_diffusion_db_train_test_valid_dataset(db_type="2m_random_10k", img_size=(224, 224))
    diffusion_db['train'].set_format(type='torch', columns=['pixel_values', 'prompt'])
    diffusion_db['test'].set_format(type='torch', columns=['pixel_values', 'prompt'])
    train_data_loader = DataLoader(dataset=diffusion_db['train'], batch_size=BATCH_SIZE)
    valid_data_loader = DataLoader(dataset=diffusion_db['test'], batch_size=BATCH_SIZE)

    sentence_transformer_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vision_captioning_model = load_pretrained_model(device=device)

    optimizer = optim.Adam(vision_captioning_model.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-4)
    optimizer.zero_grad()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    avg_train_loss_history = []
    avg_valid_loss_history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"==============================================")
        print(f"EPOCH: {epoch}")
        print(f"==============================================")
        best_sim_loss = 0
        best_sim_epoch = 0

        _, avg_train_loss = train(
            sentence_transformer_model=sentence_transformer_model,
            vision_captioning_model=vision_captioning_model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        avg_train_loss_history.append(avg_train_loss)

        _, avg_valid_loss = evaluate(
            sentence_transformer_model=sentence_transformer_model,
            vision_captioning_model=vision_captioning_model,
            data_loader=valid_data_loader,
            device=device,
        )
        avg_valid_loss_history.append(avg_valid_loss)

        print(f"Epoch {epoch} Avg. Training Loss: {avg_train_loss:.5f}")
        print(f"Epoch {epoch} Avg. Validation Loss: {avg_valid_loss:.5f}")

        if avg_valid_loss < best_sim_loss:
            best_sim_loss = avg_valid_loss
            best_sim_epoch = epoch

            if SAVE_MODEL_CHECKPOINT:
                torch.save(vision_captioning_model.state_dict(), f"cs7643_model_{timestamp}.pt")
            if SAVE_OPTIM_CHECKPOINT:
                torch.save(optimizer.state_dict(), f"cs7643_optim_{timestamp}.pt")

        if epoch - 3 > best_sim_epoch:
            print(f"Early stop condition reached.")
            print(f"Best Epoch: {best_sim_epoch}.")
            print(f"Best Cosine Similarity Score: {-best_sim_loss:.5f}.")
            break

    plot_avg_train_valid_loss(
        avg_train_cosine_loss_history=avg_train_loss_history,
        avg_valid_cosine_loss_history=avg_valid_loss_history,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="CS7643 Best Group Model Training script.")
    args = parser.parse_args()
    main(**vars(args))
