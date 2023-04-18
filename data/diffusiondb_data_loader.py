from typing import Literal, get_args
from datasets import load_dataset, DatasetDict

DATASET_SHUFFLE_SEED = 42
TEST_SET_SIZE = 0.1

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
    test_set_size: float = TEST_SET_SIZE,
) -> DatasetDict:
    options = get_args(_DIFFUSION_DB_TYPES)
    assert db_type in options, f"'{db_type}' is not in {options}"

    diffusion_db = load_dataset("poloclub/diffusiondb", db_type)

    diffusion_db_train_testvalid = diffusion_db['train'].train_test_split(test_size=test_set_size)
    diffusion_db_test_valid = diffusion_db_train_testvalid['test'].train_test_split(test_size=0.5)
    diffusion_db_train_test_valid_dataset = DatasetDict(
        {
            'train': diffusion_db_train_testvalid['train'],
            'test': diffusion_db_test_valid['test'],
            'valid': diffusion_db_test_valid['train'],
        }
    )

    return diffusion_db_train_test_valid_dataset


if __name__ == "__main__":
    diffusion_db = get_diffusion_db_train_test_valid_dataset()

    print(f"{len(diffusion_db['train'])=}")
    print(f"{len(diffusion_db['valid'])=}")
    print(f"{len(diffusion_db['test'])=}")

    sample = diffusion_db['train'][4]
    print("Example Sample:")
    for key, value in sample.items():
        print(f"\t{key}: {value}")

    sample['image'].show()
