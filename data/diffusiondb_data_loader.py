from datasets import load_dataset, DatasetDict

DATASET_SHUFFLE_SEED = 42
TEST_SET_SIZE = 0.1

diffusion_db = load_dataset("poloclub/diffusiondb")

diffusion_db_train_testvalid = diffusion_db['train'].train_test_split(
    test_size=TEST_SET_SIZE,
    shuffle=True,
    seed=DATASET_SHUFFLE_SEED
)
diffusion_db_test_valid = diffusion_db_train_testvalid['test'].train_test_split(test_size=0.5)
diffusion_db_train_test_valid_dataset = DatasetDict(
    {
        'train': diffusion_db_train_testvalid['train'],
        'test': diffusion_db_test_valid['test'],
        'valid': diffusion_db_test_valid['train'],
    }
)


if __name__ == "__main__":
    print(diffusion_db['train'][0])
    sample_image = diffusion_db['train'][0]['image']
    sample_prompt = diffusion_db['train'][0]['prompt']
    sample_image.show()
    print(f"{sample_prompt=}")

    print(f"{len(diffusion_db_train_test_valid_dataset['train'])=}")
    print(f"{len(diffusion_db_train_test_valid_dataset['valid'])=}")
    print(f"{len(diffusion_db_train_test_valid_dataset['test'])=}")
