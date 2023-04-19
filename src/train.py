from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataHelpers.coco_data_loader import get_coco_train_dataset, get_coco_valid_dataset
from dataHelpers.diffusiondb_data_loader import get_diffusion_db_train_test_valid_dataset
from image_cap import ImageCap
from loss import cosine_similarity_loss


# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1.0e-3

# Constants
VALID_DATASETS = ['coco', 'diffusion_db']


def train(dataset: str = 'coco'):
    """Main CS7643 awesomest group training loop.

    :param dataset: Which dataset to train on ('coco' or 'diffusion_db'), defaults to 'coco'
    :type dataset: str, optional
    :raises ValueError: when dataset is not one of 'coco' or 'diffusion_db'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset == 'coco':
        coco_train = get_coco_train_dataset(do_transform_from_PIL_to_tensor=True)
        coco_valid = get_coco_valid_dataset(do_transform_from_PIL_to_tensor=True)
        train_data_loader = DataLoader(dataset=coco_train, batch_size=BATCH_SIZE, collate_fn=lambda x: x )
        valid_data_loader = DataLoader(dataset=coco_valid, batch_size=BATCH_SIZE, collate_fn=lambda x: x )

    elif dataset == 'diffusion_db':
        diffusion_db = get_diffusion_db_train_test_valid_dataset(db_type="2m_random_10k")
        diffusion_db.set_format(type='torch', columns=['pixel_values', 'prompt'])
        train_data_loader = DataLoader(dataset=diffusion_db['train'], batch_size=BATCH_SIZE)
        valid_data_loader = DataLoader(dataset=diffusion_db['valid'], batch_size=BATCH_SIZE)

    else:
        raise ValueError(f"'{dataset}' not valid. Must be one of {VALID_DATASETS}.")

    model = ImageCap()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1.0e-4)
    optimizer.zero_grad()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(EPOCHS):
        print(f"==============================================")
        print(f"EPOCH: {epoch}")
        print(f"==============================================")
        epoch_loss = 0

        for batch_id, batch in enumerate(tqdm(train_data_loader)):
            print(batch)


if __name__ == "__main__":
    parser = ArgumentParser(description="CS7643 Best Group Model Training script.")
    parser.add_argument(
        '--dataset',
        type=str,
        default=VALID_DATASETS[0],
        choices=VALID_DATASETS,
        help="Which dataset to train on."
    )

    args = parser.parse_args()

    train(**vars(args))
