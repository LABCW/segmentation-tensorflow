import imgaug
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from utils.default import _C as config
from utils.default import update_config
import argparse

# !!! 必须在外部测试好数据增强
def aug_data(config, image, labels):
    seq = iaa.Sequential([   
        iaa.Sometimes(0.5, iaa.CropToSquare()),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.5, iaa.Rot90((1, 3), keep_size=False)),
        iaa.Sometimes(0.5, iaa.AddToBrightness((-20, 20))),  
        iaa.Sometimes(0.5,iaa.CropAndPad(percent=(0, 0.3),
                                        pad_mode=["constant", "edge", "linear_ramp", "maximum", "mean", "median","minimum"],
                                        pad_cval=(0,255))),
        # iaa.JpegCompression(compression=(95, 100)),
        iaa.Resize({"height": config.MODEL.HEIGHT, "width": config.MODEL.WIDTH},interpolation=('linear','cubic')),
        ], random_order=True)

    seq_det = seq.to_deterministic()
    img_aug = seq_det(image=image)
    label_aug = []
    for label in labels:
        segg = SegmentationMapsOnImage(label, shape=label.shape)
        label_out = seq_det.augment_segmentation_maps([segg])[0].get_arr()
        label_aug.append(label_out)
    return img_aug, label_aug

if __name__ == '__main__':
    import os, random, cv2
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="config.yaml", help='config')
    args = parser.parse_args()
    update_config(config, args)
    print("---------config-----------")
    print(config)
    print("--------------------------")
    
    image_train_dir = os.path.join(config.DATASET.ROOT, config.DATASET.image_train_dir)  # 真实样本，完整的label
    mask_train_dir = os.path.join(config.DATASET.ROOT, config.DATASET.mask_train_dir)
    num_classes = config.DATASET.NUM_CLASSES
    CLASSES_NAME = config.DATASET.CLASSES_NAME
    
    train_input_names = os.listdir(image_train_dir)
    random.shuffle(train_input_names)

    for base_name in train_input_names:
        image_path = os.path.join(image_train_dir, base_name)
        print('--------------------------')
        print(image_path)
        im_in = cv2.imread(image_path)

        labels_in = []
        for n in range(num_classes):
            mask_path = os.path.join(mask_train_dir, base_name.split('.')[0] + '_%s' % CLASSES_NAME[n] + config.DATASET.mask_suffix)
            img_mask = cv2.imread(mask_path, flags=0)
            labels_in.append(img_mask)

        im_out, labels_out = aug_data(config, im_in, labels_in)    
        cv2.namedWindow("img",0)
        cv2.imshow("img", im_out)
        for n in range(num_classes):
            cv2.namedWindow("mask_%s"%CLASSES_NAME[n],0)
            cv2.imshow("mask_%s"%CLASSES_NAME[n], labels_out[n])
            out = im_out.copy()
            for i in range(out.shape[0]):  # i for h
                for j in range(out.shape[1]):
                    if labels_out[n][i,j] == 255:
                        # out[i, j, 0] = color[0]
                        # out[i, j, 1] = color[1]
                        value = out[i, j, 2]+100
                        if value >= 255:
                            out[i, j, 0] = 255-100
                            out[i, j, 1] = 255-100
                            out[i, j, 2] = 255
                        else:
                            out[i, j, 2] =  value
            cv2.namedWindow("out_vis_%s"%CLASSES_NAME[n],0)
            cv2.imshow("out_vis_%s"%CLASSES_NAME[n], out)
        print("img size: ", im_in.shape, " -> ", im_out.shape)
        key = cv2.waitKey(0)  
        if key == ord('q'):
            exit()

        # # 原始图数据增强
        # seq_det = seq.to_deterministic()
        # img_aug = seq_det(image=im_in)
        # cv2.namedWindow("img",0)
        # cv2.imshow("img", img_aug)

        # for n in range(num_classes):
        #     mask_path = os.path.join(mask_train_dir, base_name.split('.')[0] + '_%s' % CLASSES_NAME[n] + config.DATASET.mask_suffix)
        #     img_mask = cv2.imread(mask_path, flags=0)
        #     # 分割mask图数据增强
        #     segg = SegmentationMapsOnImage(img_mask, shape=img_mask.shape)
        #     img_out = seq_det.augment_segmentation_maps([segg])[0].get_arr()
        #     cv2.namedWindow("mask_%s"%CLASSES_NAME[n],0)
        #     cv2.imshow("mask_%s"%CLASSES_NAME[n], img_out)
        #     # segg_roi = SegmentationMapsOnImage(img_mask_roi, shape=img.shape)
        #     # img_out_roi = seq_det.augment_segmentation_maps([segg_roi])[0].get_arr()
        
        #     out = img_aug.copy()
        #     for i in range(out.shape[0]):  # i for h
        #         for j in range(out.shape[1]):
        #             if img_out[i,j] == 255:
        #                 # out[i, j, 0] = color[0]
        #                 # out[i, j, 1] = color[1]
        #                 value = out[i, j, 2]+100
        #                 if value >= 255:
        #                     out[i, j, 0] = 255-100
        #                     out[i, j, 1] = 255-100
        #                     out[i, j, 2] = 255
        #                 else:
        #                     out[i, j, 2] =  value
        #     cv2.namedWindow("out_vis_%s"%CLASSES_NAME[n],0)
        #     cv2.imshow("out_vis_%s"%CLASSES_NAME[n], out)

        # print("img size: ", im_in.shape, " -> ", img_aug.shape)
        # key = cv2.waitKey(0)  
        # if key == ord('q'):
        #     exit()