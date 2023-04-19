import torchvision.datasets as dset
import torchvision.transforms as transforms


COCO_DATASET_TRAIN_PATH ="./data/coco/train2017"
COCO_ANNOTATIONS_TRAIN_PATH = "./data/coco/annotations/captions_train2017.json"

COCO_DATASET_VALID_PATH ="./data/coco/val2017"
COCO_ANNOTATIONS_VALID_PATH = "./data/coco/annotations/captions_val2017.json"


def get_coco_train_dataset(
    img_size: tuple = (512, 512),
    do_transform_from_PIL_to_tensor: bool = True
) -> dset.CocoCaptions:
    """_summary_

    :param img_size: _description_, defaults to (512, 512)
    :type img_size: tuple, optional
    :param do_transform_from_PIL_to_tensor: _description_, defaults to True
    :type do_transform_from_PIL_to_tensor: bool, optional
    :return: _description_
    :rtype: dset.CocoCaptions
    """
    transform_list = [transforms.Resize(img_size)]
    if do_transform_from_PIL_to_tensor:
        transform_list.append(transforms.PILToTensor())

    coco_train = dset.CocoCaptions(
        root=COCO_DATASET_TRAIN_PATH,
        annFile=COCO_ANNOTATIONS_TRAIN_PATH,
        transform=transforms.Compose(transform_list),
    )

    return coco_train

def get_coco_valid_dataset(
    img_size: tuple = (512, 512),
    do_transform_from_PIL_to_tensor: bool = True,
) -> dset.CocoCaptions:
    """_summary_

    :param img_size: _description_, defaults to (512, 512)
    :type img_size: tuple, optional
    :param do_transform_from_PIL_to_tensor: _description_, defaults to True
    :type do_transform_from_PIL_to_tensor: bool, optional
    :return: _description_
    :rtype: dset.CocoCaptions
    """
    transform_list = [transforms.Resize(img_size)]
    if do_transform_from_PIL_to_tensor:
        transform_list.append(transforms.PILToTensor())

    coco_valid = dset.CocoCaptions(
        root=COCO_DATASET_VALID_PATH,
        annFile=COCO_ANNOTATIONS_TRAIN_PATH,
        transform=transforms.Compose(transform_list),
    )

    return coco_valid


if __name__ == "__main__":
    coco_train = get_coco_train_dataset(do_transform_from_PIL_to_tensor=False)

    print(f"{len(coco_train)=}")
    img, target = coco_train[10]
    img.show()
    print(f"{target=}")
