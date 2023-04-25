import matplotlib.pyplot as plt


def plot_avg_train_valid_loss(
    avg_train_cosine_loss_history: list,
    avg_valid_cosine_loss_history: list,
):
    assert len(avg_train_cosine_loss_history) == len(avg_valid_cosine_loss_history)
    epochs = [i for i in range(1, len(avg_valid_cosine_loss_history))]

    _, ax = plt.subplots()
    ax.plot(epochs, avg_train_cosine_loss_history, label='Training Cosine Similarity Loss', marker='o')
    ax.plot(epochs, avg_valid_cosine_loss_history, label='Validation Cosine Similarity Loss', marker='o')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity Loss")
    ax.set_xticks(epochs[::10])
    ax.set_title("Image Captioning Semantic Loss Curve")
    ax.legend()
    plt.show()
    plt.savefig("CS7643_loss_curve.png")
    print("DONE")
