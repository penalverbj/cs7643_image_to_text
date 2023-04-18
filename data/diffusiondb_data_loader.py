from datasets import load_dataset


diffusion_db = load_dataset("poloclub/diffusiondb")

if __name__ == "__main__":
    diffusion_db = load_dataset("poloclub/diffusiondb")
    print(diffusion_db['train'][0])
    sample_image = diffusion_db['train'][0]['image']
    sample_prompt = diffusion_db['train'][0]['prompt']
    sample_image.show()
    print(f"{sample_prompt=}")
