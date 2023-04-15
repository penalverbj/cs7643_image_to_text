# import matplotlib.pyplot as plt
# from PIL import Image
import numpy as np
from cocoParser import cocoParser

class cocoLoader:
    def __init__(self, ann_path, imgs_path):
        self.coco_annotations_file=ann_path
        self.coco_images_dir=imgs_path
        self.parser = cocoParser(self.coco_annotations_file, self.coco_images_dir)
        self.current_idx = 0 # temporary, should probably do it some other way

    def get_imgs(self, num, random=False):
        img_ids = self.parser.get_imgIds()
        if random:
            total_images = len(img_ids) # total number of images
            img_idxs = np.random.permutation(total_images)[:num]
        else:
            img_idxs = img_ids[self.current_idx:self.current_idx + num]
            self.current_idx += num

        selected_img_ids = [img_ids[i] for i in img_idxs]
        ann_ids = self.parser.get_annIds(selected_img_ids)
        im_licenses = self.parser.get_imgLicenses(selected_img_ids)
        imgs_out = []

        for i, im in enumerate(selected_img_ids):
            image = f"{self.coco_images_dir}/{str(im).zfill(12)}.jpg"
            ann_ids = self.parser.get_annIds(im)
            annotations = self.parser.load_anns(ann_ids)

            temp = {
                "image" : image,
                "ann_ids" : ann_ids,
                "annotations" : annotations
            }
            imgs_out.append(temp)

        return imgs_out


def main():
    loader = cocoLoader("C:/Users/penal/DeepLearning/final/data\coco/annotations/captions_train2017.json", "data/coco/train2017")
    imgs = loader.get_imgs(1, random=True)
    print(imgs[0])

if __name__ == "__main__":
    main()


# num_imgs_to_disp = 4
# total_images = len(coco.get_imgIds()) # total number of images
# sel_im_idxs = np.random.permutation(total_images)[:num_imgs_to_disp]
# img_ids = coco.get_imgIds()
# selected_img_ids = [img_ids[i] for i in sel_im_idxs]
# ann_ids = coco.get_annIds(selected_img_ids)
# im_licenses = coco.get_imgLicenses(selected_img_ids)
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
# ax = ax.ravel()
# for i, im in enumerate(selected_img_ids):
#     image = Image.open(f"{self.coco_images_dir}/{str(im).zfill(12)}.jpg")
#     ann_ids = coco.get_annIds(im)
#     annotations = coco.load_anns(ann_ids)
#     for ann in annotations:
#         bbox = ann['bbox']
#         x, y, w, h = [int(b) for b in bbox]
#         class_id = ann["category_id"]
#         class_name = coco.load_cats(class_id)[0]["name"]
#         license = coco.get_imgLicenses(im)[0]["name"]
#         color_ = color_list[class_id]
#         rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_, facecolor='none')
#         t_box=ax[i].text(x, y, class_name,  color='red', fontsize=10)
#         t_box.set_bbox(dict(boxstyle='square, pad=0',facecolor='white', alpha=0.6, edgecolor='blue'))
#         ax[i].add_patch(rect)
    
#     ax[i].axis('off')
#     ax[i].imshow(image)
#     ax[i].set_xlabel('Longitude')
#     ax[i].set_title(f"License: {license}")
# plt.tight_layout()
# plt.show()