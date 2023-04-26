from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt


def plot_avg_train_valid_loss(
    avg_train_cosine_loss_history: list,
    avg_valid_cosine_loss_history: list,
    timestamp: str = None,
):
    assert len(avg_train_cosine_loss_history) == len(avg_valid_cosine_loss_history)
    epochs = [i for i in range(1, len(avg_valid_cosine_loss_history) + 1)]

    fig, ax = plt.subplots()
    ax.plot(epochs, avg_train_cosine_loss_history, label='Training Cosine Similarity Loss', marker='o')
    ax.plot(epochs, avg_valid_cosine_loss_history, label='Validation Cosine Similarity Loss', marker='o')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity Loss")
    ax.set_title("Image Captioning Semantic Loss Curve")
    ax.legend()
    plt.show()

    if timestamp is not None:
        filename = f"CS7643_loss_curve_{timestamp}.png"
    else:
        filename = "CS7643_loss_curve.png"
    fig.savefig(filename)


if __name__ == "__main__":
    parser = ArgumentParser(description="CS7643 Best Group Model Plotting script.")
    parser.add_argument(
        "--training_history",
        type=str,
        required=True,
        help="Pickle file with training history dict.",
    )
    args = parser.parse_args()
    
    # NOTE: this is highly coupled with how the training history is being pickled in train.py.
    # It will be searching for a timestamp.
    filepath = str(args.training_history)
    timestamp = filepath.split("_")[-1]
    with open(args.training_history, 'rb') as handle:
        training_history = pickle.load(handle)

    plot_avg_train_valid_loss(
        timestamp=timestamp,
        avg_train_cosine_loss_history=training_history['avg_train_loss_history'],
        avg_valid_cosine_loss_history=training_history['avg_valid_loss_history'],
    )
