from __future__ import print_function, division
import os,time,cv2, sys, math
import tensorflow as tf
import numpy as np
import time, datetime
import os, random
import helpers
# from utils import helpers

COLOUR_LIST = [[0, 0, 0], [64, 128, 64], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0], [64, 0, 128], [64, 0, 192],
     [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192], [192, 0, 64], [128, 128, 64], [192, 0, 192],
     [128, 64, 64], [64, 192, 128], [64, 64, 0], [128, 64, 128], [128, 128, 192], [0, 0, 192], [192, 128, 128],
     [128, 128, 128], [64, 128, 192], [0, 0, 64], [0, 64, 64], [192, 64, 128], [128, 128, 0], [192, 128, 192],
     [64, 0, 64], [192, 192, 0], [64, 192, 0]]

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


# Subtracts the mean images from ImageNet
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def list_images(path, file_type='images'):
    """
    列出文件夹中所有的文件，返回
    :param file_type: 'images' or 'any'
    :param path: a directory path, like '../data/pics'
    :return: all the images in the directory
    """
    IMAGE_SUFFIX = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.Png', '.PNG', '.tiff', '.bmp', '.tif']
    # IMAGE_SUFFIX = ['.png']
    paths = []
    for file_and_dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_and_dir)):
            if file_type == 'images':
                if os.path.splitext(file_and_dir)[1] in IMAGE_SUFFIX:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
            elif file_type == 'any':
                paths.append(os.path.abspath(os.path.join(path, file_and_dir)))
            else:
                if os.path.splitext(file_and_dir)[1] == file_type:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
    return paths


def rotate(image, angle=90):
    height, width = image.shape[:2]
    degree = angle
    heightNew = int(width * math.fabs(math.sin(math.radians(degree))) +
                    height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(height * math.fabs(math.sin(math.radians(degree))) +
                   width * math.fabs(math.cos(math.radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    return imgRotation

def resize_image(image, size, interpolation='INTER_LINEAR'):
    if image.shape[0] > size[1]:
        resize_ratio = size[1] / float(image.shape[0])
        if interpolation == 'INTER_LINEAR':
            image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)
    if image.shape[1] > size[0]:
        resize_ratio = size[0] / float(image.shape[1])
        if interpolation == 'INTER_LINEAR':
            image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)
    return image, None


def random_rotate_and_flip(image, labels, rand_threhold=0.3):
    """
    image: 原图
    labels: list,根据分类数量可能有多个label
    """
    # 随机旋转
    rand_val = random.random()
    if rand_val < rand_threhold:
        angle = random.choice([90, 180, 270])
        image = rotate(image, angle=angle)
        labels = [rotate(label, angle=angle) for label in labels]

    # 随机水平翻转
    if random.randint(0, 1):
        image = cv2.flip(image, 1)
        labels = [cv2.flip(label, 1) for label in labels]

    # 随机垂直翻转
    if random.randint(0, 1):
        image = cv2.flip(image, 0)
        labels = [cv2.flip(label, 0) for label in labels]

    # 二值化，确保label是二值的
    thre_labels = []
    for label in labels:
        _, thre_label = cv2.threshold(label, 128, 255, cv2.THRESH_BINARY)
        thre_labels.append(thre_label)

    return image, thre_labels


def random_stretch(image, labels, rand_threhold=0.3):
    rand_val = random.random()
    if rand_val < rand_threhold:
        # 随机拉伸
        stretch_ratio = random.uniform(0.75, 1.25)
        rand_w = image.shape[1]
        rand_h = image.shape[0]
        if random.randint(0, 1):
            rand_w = int(rand_w * stretch_ratio)
        else:
            rand_h = int(rand_h * stretch_ratio)

        image = cv2.resize(image, (rand_w, rand_h))
        labels = [cv2.resize(label, (rand_w, rand_h)) for label in labels]

    return image, labels


def random_resize_and_paste(image, labels, size, rand_threhold=0.25, corner_threhold=0.7):
    w_init = size[0]
    h_init = size[1]
    rand_val = random.random()
    if rand_val < rand_threhold:
        # 对样本进行一定概率的随机尺寸缩放
        rand_w = np.random.randint(int(w_init * 0.7), w_init)
        rand_h = np.random.randint(int(h_init * 0.7), h_init)
        image = cv2.resize(image, (rand_w, rand_h))
        labels = [cv2.resize(label, (rand_w, rand_h)) for label in labels]

    h = image.shape[0]
    w = image.shape[1]

    # 计算图像粘贴的起始点范围
    range_x = w_init - w - 1
    range_y = h_init - h - 1

    if range_x < 0:
        range_x = 0
    if range_y < 0:
        range_y = 0

    # 随机生成起始点
    rand_x = random.randint(0, range_x)
    rand_y = random.randint(0, range_y)

    # 按照一定的概率比例，将训练数据粘贴到空白图像的4个角
    pos_list = [(0, 0), (0, range_x), (range_y, 0), (range_y, range_x)]
    rand_index = random.randint(0, 3)
    rand_pb = random.random()  # 随机产生一个概率
    if rand_pb < corner_threhold:
        rand_y = pos_list[rand_index][0]
        rand_x = pos_list[rand_index][1]

    # 生成空白图像（全白色），用于装载原始图像
    # 彩色图改为生成随机彩图
    if image.shape[2] == 3:
        paste_image = np.ones([h_init, w_init, 3], dtype=np.uint8) * 255
        # randomByteArray = bytearray(os.urandom(w_init*h_init*3))
        # flatNumpyArray = np.array(randomByteArray)
        # paste_image = flatNumpyArray.reshape(h_init, w_init, 3)
        paste_image[rand_y:h + rand_y, rand_x:w + rand_x, :] = image[:, :, :]
    else:
        paste_image = np.ones([h_init, w_init], dtype=np.uint8) * 255
        paste_image[rand_y:h + rand_y, rand_x:w + rand_x] = image[:, :]

    new_labels = []
    for n in range(len(labels)):
        # 生成空白图像（全黑色），用于装载前景掩模图像
        score_map = np.zeros([h_init, w_init], dtype=np.uint8)
        score_map[rand_y:h + rand_y, rand_x:w + rand_x] = labels[n][:, :]
        new_labels.append(score_map)

    return paste_image, new_labels


def random_jpg_quality(input_image, CHANNEL_NUM=3):
    # 压缩原图
    if random.randint(0, 1):
        jpg_quality = np.random.randint(60, 100)
        _, input_image = cv2.imencode('.jpg', input_image, [1, jpg_quality])
        if CHANNEL_NUM == 3:
            input_image = cv2.imdecode(input_image, 1)
        else:
            input_image = cv2.imdecode(input_image, 0)

    return input_image

def random_bright(input_image):
    # 随机亮度
    if random.randint(0, 1):
        factor = 1.0 + random.uniform(-0.4, 0.4)
        table = np.array([min((i * factor), 255.) for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)

    return input_image

def focal_loss_weight_map(y_true_cls, y_pred_cls, n_class):
    # 生成focal loss的weight map
    gamma = 2.
    alpha = 0.5

    # 展平
    flat_y_true_cls = tf.reshape(y_true_cls, [-1, n_class])
    flat_y_pred_cls = tf.reshape(y_pred_cls, [-1, n_class])

    # 分离通道
    flat_y_true_cls_split = tf.split(flat_y_true_cls, n_class, axis=1)
    flat_y_pred_cls_split = tf.split(flat_y_pred_cls, n_class, axis=1)

    pt_list = list()
    for n in range(n_class):
        pt = flat_y_true_cls_split[n] * flat_y_pred_cls_split[n] + \
              (1.0 - flat_y_true_cls_split[n]) * (1.0 - flat_y_pred_cls_split[n])
        pt_list.append(pt)

    weight_map_list = list()
    for n in range(n_class):
        weight_map_list.append(alpha * tf.pow((1.0 - pt_list[n]), gamma))

    # 拼接通道
    if n_class == 1:
        weighted_map = weight_map_list[0]
    else:
        weighted_map = tf.concat(tuple(weight_map_list), axis=1)

    return weighted_map