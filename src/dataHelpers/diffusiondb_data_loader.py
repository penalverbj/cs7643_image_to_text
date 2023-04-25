from typing import Literal, get_args
from datasets import load_dataset, DatasetDict
from transformers import CLIPProcessor


DATASET_SHUFFLE_SEED = 42

_DIFFUSION_DB_TYPES = Literal[
    "2m_random_1k",
    "2m_random_5k",
    "2m_random_10k",
    "2m_random_50k",
    "2m_random_100k",
    "2m_random_500k",
    "2m_random_1m",
    "2m_all",
    "large_random_1k",
    "large_random_5k",
    "large_random_10k",
    "large_random_50k",
    "large_random_100k",
    "large_random_500k",
    "large_random_1m",
    "large_all",
]


def get_diffusion_db_train_test_valid_dataset(
    db_type: _DIFFUSION_DB_TYPES = "2m_random_1k",
    test_set_size: float = 0.1,
    img_size: tuple = (224, 224),
) -> DatasetDict:
    """Gets a diffusion_db training, validation, and test datset as a HuggingFace DatasetDict
    of uniform image size.
    https://huggingface.co/docs/datasets/v1.1.1/package_reference/main_classes.html#datasetdict

    For use in a PyTorch `torch.DataLoader`, see this resource:
    https://huggingface.co/docs/datasets/v1.3.0/torch_tensorflow.html

    Call the built-in `set_format` method to bring the data into Tensor format.

    See the link below for more details on the dataset and the options for db_type.
    https://huggingface.co/datasets/poloclub/diffusiondb

    :param db_type: Which subset of diffusion_db to pull from huggingface, defaults to "2m_random_1k"
    :type db_type: _DIFFUSION_DB_TYPES, optional
    :param test_set_size: Fraction of the dataset (b/w 0 and 1) to use in test/validation, defaults to 0.1
    :type test_set_size: float, optional
    :param img_size: image size in (w, h) to make all images uniform, defaults to (512, 512)
    :type img_size: tuple, optional
    :return: DatasetDict with 'train', 'valid', and 'test' keys.
    :rtype: DatasetDict
    """
    options = get_args(_DIFFUSION_DB_TYPES)
    assert db_type in options, f"'{db_type}' is not in {options}"

    diffusion_db = load_dataset("poloclub/diffusiondb", db_type)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(examples):
        images = [image.convert("RGB").resize(img_size) for image in examples["image"]]
        examples['pixel_values'] = [clip_processor(images=image) for image in images]
        return examples

    preprocessed_diffusion_db = diffusion_db['train'].map(encode_image, remove_columns=["image"], batched=True)
    diffusion_db_train_valid = preprocessed_diffusion_db.train_test_split(test_size=test_set_size)

    return diffusion_db_train_valid


if __name__ == "__main__":
    diffusion_db = get_diffusion_db_train_test_valid_dataset()

    print(f"{len(diffusion_db['train'])=}")
    print(f"{len(diffusion_db['valid'])=}")
    print(f"{len(diffusion_db['test'])=}")

    sample = diffusion_db['train'][4]
    print("Example Sample:")
    for key, value in sample.items():
        print(f"\t{key}: {value}")

    sample['pixel_values'].show()
