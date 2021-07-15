#!/usr/bin/env python2
# coding: utf-8
from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys

from utils import utils
from utils.utils import COLOUR_LIST
from utils.helpers import colour_code_segmentation, one_hot_it
from builders import model_builder
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

IMAGE_SIZE = 768
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--dataset', type=str, default="sample", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=IMAGE_SIZE, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=IMAGE_SIZE, help='Width of cropped input image to network')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=0.5, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--model', type=str, default="DeepLabV3_plus", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
args = parser.parse_args()

g_step = 0
num_classes = 2  # 类别数量，多分类需要背景类
CHANNEL_NUM = 3  # 'rgb' or 'gray'
learning_rate = 0.00001
BATCH_SIZE = 1
have_ok_image = True
read_ok = 3
checkpoint_path = '/home/root12/jinsong/miniled_training_data/checkpoint_DeepLabV3_plus'
use_focal_loss = False  # 是否用focal loss
use_class_balance_weights = False
class_name = ['0', '1']  # 名字与mask的命名要对应,背景类不需要写，要按照优先级排列，优先级高的类型放后面
class_model = 'multi_label'  # 配置多分类和多标签：multi_classify、multi_label
label_gray_diff = 40  # 多分类的灰度差，用于tensorboard显示
image_train_dir = '/home/root12/jinsong/miniled_training_data/ng'  # 真实样本，完整的label
mask_train_dir = '/home/root12/jinsong/miniled_training_data/label'
# ok_image_dir = '/home/root12/jinsong/miniled_training_data/ok'
# crop_dir = '/home/xddz/workspace/zw/chip_training_data/croped_image'
crop_dir = None
# crop_rand_threhold = 0.3
ok_root_dir = '/home/root12/jinsong/miniled_training_data/ok'
ok_file = ['ok', 'nothing']

# 给每个类一个colour用于生成onehot
label_values = list()
for i in range(num_classes):
    label_values.append(COLOUR_LIST[i])


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


def resize_image(image, max_edge=IMAGE_SIZE, interpolation='INTER_LINEAR'):
    # 如果最大边长大于max_edge，限制最大边长到max_edge
    max_size = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
    if max_size > max_edge:
        resize_ratio = max_edge / float(max_size)
        if interpolation == 'INTER_LINEAR':
            image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)
    return image, None


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

    # 样本越易分，pt越接近1，则贡献的loss就越小
    # 样本越难分，pt越接近0，则贡献的loss就越小
    # pt0 = flat_y_true_cls_split[0] * flat_y_pred_cls_split[0] + \
    #       (1.0 - flat_y_true_cls_split[0]) * (1.0 - flat_y_pred_cls_split[0])
    #
    # pt1 = flat_y_true_cls_split[1] * flat_y_pred_cls_split[1] + \
    #       (1.0 - flat_y_true_cls_split[1]) * (1.0 - flat_y_pred_cls_split[1])
    #
    # pt2 = flat_y_true_cls_split[2] * flat_y_pred_cls_split[2] + \
    #       (1.0 - flat_y_true_cls_split[2]) * (1.0 - flat_y_pred_cls_split[2])

    pt_list = list()
    for n in range(n_class):
        pt = flat_y_true_cls_split[n] * flat_y_pred_cls_split[n] + \
              (1.0 - flat_y_true_cls_split[n]) * (1.0 - flat_y_pred_cls_split[n])
        pt_list.append(pt)

    # 易分的样本，pt接近1，tf.pow((1.0 - pt0), gamma)的结果就非常接近0
    # 难分的样本，pt接近0，tf.pow((1.0 - pt0), gamma)的结果就非常接近1
    # focal loss 就是通过分割的结果与gt进行比较，区分出容易分割和不容易分割的像素，然后有针对性的分配权重
    # weight_map_0 = alpha * tf.pow((1.0 - pt0), gamma)
    # weight_map_1 = alpha * tf.pow((1.0 - pt1), gamma)
    # weight_map_2 = alpha * tf.pow((1.0 - pt2), gamma)

    weight_map_list = list()
    for n in range(n_class):
        weight_map_list.append(alpha * tf.pow((1.0 - pt_list[n]), gamma))

    # 拼接通道
    if n_class == 1:
        weighted_map = weight_map_list[0]
    else:
        weighted_map = tf.concat(tuple(weight_map_list), axis=1)

    return weighted_map

# def focal_loss_weight_map(y_true_cls, y_pred_cls, n_class):
#     # 生成focal loss的weight map
#     gamma = 2.
#     alpha = 0.5
#
#     # 展平
#     flat_y_true_cls = tf.reshape(y_true_cls, [-1, n_class])
#     flat_y_pred_cls = tf.reshape(y_pred_cls, [-1, n_class])
#
#     # 分离通道
#     flat_y_true_cls_split = tf.split(flat_y_true_cls, n_class, axis=1)
#     flat_y_pred_cls_split = tf.split(flat_y_pred_cls, n_class, axis=1)
#
#     # 样本越易分，pt越接近1，则贡献的loss就越小
#     # 样本越难分，pt越接近0，则贡献的loss就越小
#     pt0 = flat_y_true_cls_split[0] * flat_y_pred_cls_split[0] + \
#         (1.0 - flat_y_true_cls_split[0]) * (1.0 - flat_y_pred_cls_split[0])
#
#     # 易分的样本，pt接近1，tf.pow((1.0 - pt0), gamma)的结果就非常接近0
#     # 难分的样本，pt接近0，tf.pow((1.0 - pt0), gamma)的结果就非常接近1
#     # focal loss 就是通过分割的结果与gt进行比较，区分出容易分割和不容易分割的像素，然后有针对性的分配权重
#     weight_map_0 = alpha * tf.pow((1.0 - pt0), gamma)
#
#     # 拼接三个通道
#     # weighted_map = tf.concat((weight_map_0, weight_map_1), axis=1)
#     weighted_map = weight_map_0
#     return weighted_map


def get_loss(logits, label_input, use_focal_loss, use_class_balance_weights, num_classes, class_model):
    # 展平
    flat_logits = tf.reshape(logits, [-1, num_classes])
    flat_labels = tf.reshape(label_input, [-1, num_classes])

    if use_focal_loss:
        # 得到focal loss计算出来的weight_map
        flat_logits = tf.nn.sigmoid(flat_logits)
        weight_map = focal_loss_weight_map(flat_labels, flat_logits, num_classes)

        # 计算交叉熵损失
        loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)

        weighted_loss = tf.multiply(loss_map, weight_map)
    else:
        # 计算交叉熵损失
        if class_model == 'multi_label':
            loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
        elif class_model == 'multi_classify':
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)

        weighted_loss = loss_map

    if use_focal_loss and use_class_balance_weights:
        # 使用focal loss的时候，虽然在每个类型的通道内合理的分配了容易分割和不容易分割的权重，
        # 但没有办法解决类型通道之间的前景数量不均衡的问题，如果不增加类型通道之间的相对权重的话，图例很难分割出来，原因是图例所占的比例太低，
        # 很难主导梯度的更新方向。增加类型通道权重后，图例很快就收敛了
        # 用这个方法的前提是三个类型的label是完整的，原因是，这个权重本身就是类别通道之间的权重，如果三个类型通道不完全，
        # 那意味着有些通道的loss压根就不计算，也就没必要再用上权重了。
        class_balance_weights = np.array([[12.47, 1.0]])
        class_balance_weights = tf.constant(np.array(class_balance_weights, dtype=np.float32))
        # class_balance_weights = tf.multiply(flat_labels, class_balance_weights)
        # 偏置常量，目的是让背景的权重为1
        # bias = tf.constant([1], dtype=tf.float32)
        # class_balance_weights = tf.add(class_balance_weights, bias)
        weighted_loss = tf.multiply(weighted_loss, class_balance_weights)

    mean_loss = tf.reduce_mean(weighted_loss)

    return mean_loss


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


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


def random_resize_and_paste(image, labels, rand_threhold=0.25, corner_threhold=0.7):
    rand_val = random.random()
    if rand_val < rand_threhold:
        # 对样本进行一定概率的随机尺寸缩放
        rand_w = np.random.randint(int(IMAGE_SIZE * 0.7), IMAGE_SIZE)
        rand_h = np.random.randint(int(IMAGE_SIZE * 0.7), IMAGE_SIZE)
        image = cv2.resize(image, (rand_w, rand_h))
        labels = [cv2.resize(label, (rand_w, rand_h)) for label in labels]

    h = image.shape[0]
    w = image.shape[1]

    # 计算图像粘贴的起始点范围
    range_x = IMAGE_SIZE - w - 1
    range_y = IMAGE_SIZE - h - 1

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
    if CHANNEL_NUM == 3:
        paste_image = np.ones([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8) * 255
        paste_image[rand_y:h + rand_y, rand_x:w + rand_x, :] = image[:, :, :]
    else:
        paste_image = np.ones([IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8) * 255
        paste_image[rand_y:h + rand_y, rand_x:w + rand_x] = image[:, :]

    new_labels = []
    for n in range(num_classes):
        # 生成空白图像（全黑色），用于装载前景掩模图像
        score_map = np.zeros([IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8)
        score_map[rand_y:h + rand_y, rand_x:w + rand_x] = labels[n][:, :]
        new_labels.append(score_map)

    return paste_image, new_labels


def random_jpg_quality(input_image):
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


def random_paste_croped_image(crop_image_list, image, label, crop_rand_threhold=0.3):
    rand_val = random.random()
    if len(crop_image_list) == 0:
        return image, label

    if rand_val < crop_rand_threhold:
        crop_image_path = random.choice(crop_image_list)
        crop_image = cv2.imread(crop_image_path)

        # 随机旋转
        if random.randint(0, 1):
            angle = random.choice([90, 180, 270])
            crop_image = rotate(crop_image, angle=angle)

        # 随机水平翻转
        if random.randint(0, 1):
            crop_image = cv2.flip(crop_image, 1)

        # 随机垂直翻转
        if random.randint(0, 1):
            crop_image = cv2.flip(crop_image, 0)

        h = crop_image.shape[0]
        w = crop_image.shape[1]

        h_start = np.random.randint(1, IMAGE_SIZE-h-1)
        w_start = np.random.randint(1, IMAGE_SIZE-w-1)

        image[h_start: h_start+h, w_start: w_start+w, :] = crop_image[:]

        label[h_start: h_start+h, w_start: w_start+w] = 255

    return image, label


ok_image_list = [list_images(os.path.join(ok_root_dir, f)) for f in ok_file]

crop_image_list = list_images(crop_dir) if (crop_dir is not None) else []

# 输入的图像
net_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM])

# 输入的label
net_output = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, num_classes])

# 全局步数
global_step = tf.Variable(0)

# net_input = mean_image_subtraction(net_input)  # 图像归一化

# 构造model
logits, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input,
                                            num_classes=num_classes, crop_width=args.crop_width,
                                            crop_height=args.crop_height, dropout_p=0.0, is_training=True)
print(logits)
if class_model == 'multi_classify':
    activation_logits = tf.nn.softmax(logits)

    loss = get_loss(logits, net_output, use_focal_loss, use_class_balance_weights, num_classes, class_model)

    gt_label = tf.argmax(net_output, axis=-1)

    gt_maps = [tf.reshape(gt_label[n, :, :], (1, IMAGE_SIZE, IMAGE_SIZE, 1)) for n in range(BATCH_SIZE)]

    pred = tf.argmax(activation_logits, axis=-1)
    # 预测结果
    pred_maps = [tf.reshape(pred[n, :, :], (1, IMAGE_SIZE, IMAGE_SIZE, 1)) for n in range(BATCH_SIZE)]

    tf.summary.image('input', net_input)
    tf.summary.scalar('loss', loss)

    for n in range(BATCH_SIZE):
        gtmap = gt_maps[n] * label_gray_diff
        gtmap = tf.to_float(gtmap, name='ToFloat')
        predmap = pred_maps[n] * label_gray_diff
        predmap = tf.to_float(predmap, name='ToFloat')
        tf.summary.image('image/score_map_%s' % str(n), gtmap)
        tf.summary.image('image/score_map_pred_%s' % str(n), predmap)

elif class_model == 'multi_label':
    activation_logits = tf.nn.sigmoid(logits)

    loss = get_loss(logits, net_output, use_focal_loss, use_class_balance_weights, num_classes, class_model)

    # gt label
    gt_maps = list()
    for n in range(BATCH_SIZE):
        for i in range(num_classes):
            gt_maps.append(tf.reshape(net_output[n, :, :, i], (1, IMAGE_SIZE, IMAGE_SIZE, 1)))
    # gt_maps = [tf.reshape(net_output[0, :, :, i], (1, IMAGE_SIZE, IMAGE_SIZE, 1)) for i in range(num_classes)]

    # 预测结果
    pred_maps = list()
    for n in range(BATCH_SIZE):
        for i in range(num_classes):
            pred_maps.append(tf.reshape(activation_logits[n, :, :, i], (1, IMAGE_SIZE, IMAGE_SIZE, 1)))
    # pred_maps = [tf.reshape(activation_logits[0, :, :, i], (1, IMAGE_SIZE, IMAGE_SIZE, 1)) for i in range(num_classes)]

    for n in range(BATCH_SIZE):
        tf.summary.image('input/image_%s' % str(n), tf.reshape(net_input[n, :, :, :], (1, IMAGE_SIZE, IMAGE_SIZE, 3)))
    tf.summary.scalar('loss', loss)
    for n in range(BATCH_SIZE):

        for i in range(num_classes):
            tf.summary.image('image/%s_score_map_%s' % (class_name[i], str(n)), gt_maps[n*num_classes+i])
            tf.summary.image('image/%s_score_map_pred_%s' % (class_name[i], str(n)), pred_maps[n*num_classes+i])

# opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])
# opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, var_list=[var for var in tf.trainable_variables()])
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
saver = tf.train.Saver()
summary_op = tf.summary.merge_all()

# NG样本
train_input_names = os.listdir(image_train_dir)

# OK样本
# ok_image_names = os.listdir(ok_image_dir) if (ok_image_dir is not None) else None

print("\n***** Begin training *****")

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

summary_writer = tf.summary.FileWriter(checkpoint_path, tf.get_default_graph())

# Do the training here
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 重载历史checkpoint
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    sess.graph.finalize()

    # 对真实样本进行迭代，每迭代完一次真实样本算是一次epoch
    for epoch in range(args.epoch_start_i, args.num_epochs):
        # 打乱样本顺序
        id_list = np.random.permutation(len(train_input_names))

        # 本轮epoch需要迭代的次数==真实样本的数量

        num_iters = int(np.floor(len(id_list)))

        # 训练策略是NG样本训练5次后训练1次OK样本
        i_counter = 0
        b = 0
        neg_flag = False
        input_image_batch = []
        output_image_batch = []
        while i_counter < num_iters:
            i = i_counter
            id = id_list[i]

            # 读取image 和 label
            score_map_in = []

            if i != 0 and i % read_ok == 0 and (neg_flag is False) and have_ok_image:
                # OK
                neg_flag = True

                image_path = random.choice(ok_image_list[random.choice([x for x in range(len(ok_file))])])

                if CHANNEL_NUM == 3:
                    im_in = cv2.imread(image_path)
                else:
                    im_in = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                for n in range(num_classes):
                    score_map_in.append(np.zeros([im_in.shape[0], im_in.shape[1]], dtype=np.uint8))

                if num_classes == 1:
                    im_in, label = random_paste_croped_image(crop_image_list, im_in, score_map_in[0])
                    score_map_in = [label]

            else:
                # NG
                i_counter += 1
                neg_flag = False
                base_name = train_input_names[id]
                image_path = os.path.join(image_train_dir, base_name)

                if CHANNEL_NUM == 3:
                    im_in = cv2.imread(image_path)
                else:
                    im_in = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if class_model == 'multi_label':
                    # 多标签
                    for n in range(num_classes):
                        if '.jpg' in base_name:
                            mask_path = os.path.join(mask_train_dir,
                                                     base_name.replace('.jpg', '_%s_mask.png' % class_name[n]))
                        else:
                            mask_path = os.path.join(mask_train_dir,
                                                     base_name.replace('.png', '_%s_mask.png' % class_name[n]))
                        score_map_in.append(cv2.imread(mask_path, flags=0))

                    if num_classes == 1:
                        im_in, label = random_paste_croped_image(crop_image_list, im_in, score_map_in[0])
                        score_map_in = [label]

                elif class_model == 'multi_classify':
                    # 多分类
                    # 先给一张背景类
                    score_map_in.append(np.zeros([im_in.shape[0], im_in.shape[1]], dtype=np.uint8))

                    # 再加载缺陷类的mask,循环时需要减去背景类
                    for n in range(num_classes-1):
                        if '.jpg' in base_name:
                            mask_path = os.path.join(mask_train_dir,
                                                     base_name.replace('.jpg', '_%s_mask.png' % class_name[n]))
                        else:
                            mask_path = os.path.join(mask_train_dir,
                                                     base_name.replace('.png', '_%s_mask.png' % class_name[n]))
                        score_map_in.append(cv2.imread(mask_path, flags=0))

            # 数据增强
            im_in, score_map_in = random_rotate_and_flip(im_in, score_map_in)

            im_in, score_map_in = random_stretch(im_in, score_map_in)

            # 限制图像的最大边长，原图的双线性。label用双线性，最近邻当前景太细太小，会有像素上的损失
            im_in, _ = resize_image(im_in, interpolation='INTER_LINEAR')
            score_map_in = [resize_image(cell, interpolation='INTER_LINEAR')[0] for cell in score_map_in]

            im_in, score_map_in = random_resize_and_paste(im_in, score_map_in)

            im_in = random_jpg_quality(im_in)
            im_in = random_bright(im_in)
            # cv2.imshow('image', input_image)
            # cv2.waitKey()

            # 把图像转成浮点
            im_in = im_in.astype(np.float32)
            score_map_in = [cell.astype(np.float32) for cell in score_map_in]

            # 如果生成的掩模图像是0-255的，在这里就被转换为0-1的图像了
            for cell in score_map_in:
                cell[cell > 0.] = 1.

            # 合并通道
            if len(score_map_in) == 1:
                # 单分类
                output_image = score_map_in[0]
            else:
                if class_model == 'multi_label':
                    output_image = np.stack(np.array(score_map_in), axis=2)
                elif class_model == 'multi_classify':
                    gt = np.zeros([im_in.shape[0], im_in.shape[1], num_classes], dtype=np.uint8)
                    for n in range(1, len(score_map_in)):
                        pos = np.vstack(np.where(score_map_in[n] > 0.))
                        gt[pos[0, :], pos[1, :]] = label_values[n]
                        semantic_map = one_hot_it(gt, label_values)
                        output_image = semantic_map.astype(np.float)

            input_image = np.float32(im_in) / 255.
            input_image_batch.append(input_image)
            output_image_batch.append(output_image)
            # input_image_batch.append(np.expand_dims(input_image, axis=0))
            # output_image_batch.append(np.expand_dims(output_image, axis=0))

            if len(input_image_batch) == BATCH_SIZE:
                g_step += 1
                # 已经把batch的数量改成只能为1
                input_image_batch = np.array(input_image_batch)
                output_image_batch = np.array(output_image_batch)

                if len(output_image_batch.shape) < 4:
                    output_image_batch = np.expand_dims(output_image_batch, axis=4)

                if len(input_image_batch.shape) < 4:
                    input_image_batch = np.expand_dims(input_image_batch, axis=4)

                # Do the training
                _, current, summary_str = sess.run([opt, loss, summary_op], feed_dict={net_input: input_image_batch,
                                                                                       net_output: output_image_batch})
                if g_step % 50 == 0:
                    string_print = "g_step = %d Current_Loss = %.6f " % (g_step, current)
                    utils.LOG(string_print)

                if g_step % 50 == 0:
                    saver.save(sess, checkpoint_path + "/model.ckpt", global_step=global_step)

                if g_step % 50 == 0:
                    # g_step % 50 == 1 的时候存真实样本
                    # g_step % 50 == 0 的时候存附加样本
                    summary_writer.add_summary(summary_str, global_step=g_step)
                input_image_batch = []
                output_image_batch = []
            else:
                continue






