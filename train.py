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
from utils.utils import COLOUR_LIST, resize_image, random_rotate_and_flip,random_stretch,random_resize_and_paste,random_jpg_quality,random_bright,focal_loss_weight_map
from utils.helpers import colour_code_segmentation, one_hot_it
from builders import model_builder
from utils.default import _C as config
from utils.default import update_config

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default="config.yaml", help='config')
args = parser.parse_args()
update_config(config, args)
print("---------config-----------")
print(config)
print("--------------------------")
os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.GPUS[-1]

g_step = 0
label_gray_diff = 40  # 多分类的灰度差，用于tensorboard显示

num_classes = config.DATASET.NUM_CLASSES  # 类别数量，多分类需要背景类
CHANNEL_NUM = config.MODEL.CHANNEL_NUM  # 'rgb' or 'gray'
BATCH_SIZE = config.MODEL.BATCH
class_model = config.MODEL.class_model  # 配置多分类和多标签：multi_classify、multi_label

class_name = config.DATASET.CLASSES_NAME  # 名字与mask的命名要对应,背景类不需要写，要按照优先级排列，优先级高的类型放后面
image_train_dir = os.path.join(config.DATASET.ROOT, config.DATASET.image_train_dir)  # 真实样本，完整的label
mask_train_dir = os.path.join(config.DATASET.ROOT, config.DATASET.mask_train_dir)
ok_image_dir = config.DATASET.ok_image_dir
have_ok_image = True if ok_image_dir else False
read_ok = 3

learning_rate = config.TRAIN.learning_rate
checkpoint_path = config.TRAIN.checkpoint_path
use_focal_loss = config.TRAIN.focal_loss  # 是否用focal loss
use_class_balance_weights = True if config.TRAIN.class_balance else False

# NG样本
train_input_names = os.listdir(image_train_dir)
# OK样本
ok_image_names = os.listdir(ok_image_dir) if (ok_image_dir is not None) else None

# 输入的图像
net_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, config.MODEL.HEIGHT, config.MODEL.WIDTH, CHANNEL_NUM])
# 输入的label
net_output = tf.placeholder(tf.float32, shape=[BATCH_SIZE, config.MODEL.HEIGHT, config.MODEL.WIDTH, num_classes])
# net_input = mean_image_subtraction(net_input)  # 图像归一化
# 构造model
logits, init_fn = model_builder.build_model(model_name=config.MODEL.NAME, 
                                            frontend=config.MODEL.FRONTEND, net_input=net_input,
                                            num_classes=num_classes, 
                                            crop_width=config.MODEL.WIDTH,
                                            crop_height=config.MODEL.HEIGHT, 
                                            dropout_p=0.0, is_training=True)
print(logits)

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

if class_model == 'multi_classify':
    # 给每个类一个colour用于生成onehot
    label_values = list()
    for i in range(num_classes):
        label_values.append(COLOUR_LIST[i])

    activation_logits = tf.nn.softmax(logits)

    loss = get_loss(logits, net_output, use_focal_loss, use_class_balance_weights, num_classes, class_model)

    gt_label = tf.argmax(net_output, axis=-1)

    gt_maps = [tf.reshape(gt_label[n, :, :], (1, config.MODEL.HEIGHT, config.MODEL.WIDTH, 1)) for n in range(BATCH_SIZE)]

    pred = tf.argmax(activation_logits, axis=-1)
    # 预测结果
    pred_maps = [tf.reshape(pred[n, :, :], (1, config.MODEL.HEIGHT, config.MODEL.WIDTH, 1)) for n in range(BATCH_SIZE)]

    img_show = tf.reshape(net_input[:, :, :, :], (1, config.MODEL.HEIGHT, config.MODEL.WIDTH, 3))
    img_show = tf.reverse(img_show, axis=[-1]) 
    tf.summary.image('input', img_show)
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
            gt_maps.append(tf.reshape(net_output[n, :, :, i], (1, config.MODEL.HEIGHT, config.MODEL.WIDTH, 1)))

    # 预测结果
    pred_maps = list()
    for n in range(BATCH_SIZE):
        for i in range(num_classes):
            pred_maps.append(tf.reshape(activation_logits[n, :, :, i], (1, config.MODEL.HEIGHT, config.MODEL.WIDTH, 1)))

    for n in range(BATCH_SIZE):
        img_show = tf.reshape(net_input[n, :, :, :], (1, config.MODEL.HEIGHT, config.MODEL.WIDTH, 3))
        img_show = tf.reverse(img_show, axis=[-1]) 
        tf.summary.image('input/image_%s' % str(n), img_show)
    tf.summary.scalar('loss', loss)
    for n in range(BATCH_SIZE):

        for i in range(num_classes):
            tf.summary.image('image/%s_score_map_%s' % (class_name[i], str(n)), gt_maps[n*num_classes+i])
            tf.summary.image('image/%s_score_map_pred_%s' % (class_name[i], str(n)), pred_maps[n*num_classes+i])
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(checkpoint_path, tf.get_default_graph())

# 全局步数
global_step = tf.Variable(0)
# 定义优化器
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

print("\n***** Begin training *****")
print("---------- ", checkpoint_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
# Do the training here
saver = tf.train.Saver(max_to_keep=2)
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())

    # 重载历史checkpoint
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    sess.graph.finalize()

    # 对真实样本进行迭代，每迭代完一次真实样本算是一次epoch
    for epoch in range(config.TRAIN.epoch_start_i, config.TRAIN.num_epochs):
        epoch_st=time.time()
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
                base_name = random.choice(ok_image_names)
                image_path = os.path.join(ok_image_dir, base_name)

                if CHANNEL_NUM == 3:
                    im_in = cv2.imread(image_path)
                else:
                    im_in = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                for n in range(num_classes):
                    score_map_in.append(np.zeros([im_in.shape[0], im_in.shape[1]], dtype=np.uint8))

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
                        mask_path = os.path.join(mask_train_dir, base_name.split('.')[0] + '_%s' % class_name[n] + config.DATASET.mask_suffix)
                        score_map_in.append(cv2.imread(mask_path, flags=0))

                elif class_model == 'multi_classify':
                    # 多分类
                    # 先给一张背景类
                    score_map_in.append(np.zeros([im_in.shape[0], im_in.shape[1]], dtype=np.uint8))

                    # 再加载缺陷类的mask,循环时需要减去背景类
                    for n in range(num_classes-1):
                        mask_path = os.path.join(mask_train_dir, base_name.split('.')[0] + '_%s' % class_name[n] + config.DATASET.mask_suffix)
                        score_map_in.append(cv2.imread(mask_path, flags=0))

            # 数据增强
            im_in, score_map_in = random_rotate_and_flip(im_in, score_map_in)

            im_in, score_map_in = random_stretch(im_in, score_map_in)

            # 限制图像的最大边长，原图的双线性。label用双线性，最近邻当前景太细太小，会有像素上的损失
            im_in, _ = resize_image(im_in,size=(config.MODEL.WIDTH, config.MODEL.HEIGHT), interpolation='INTER_LINEAR')
            score_map_in = [resize_image(cell,size=(config.MODEL.WIDTH, config.MODEL.HEIGHT), interpolation='INTER_LINEAR')[0] for cell in score_map_in]

            im_in, score_map_in = random_resize_and_paste(im_in, score_map_in,size=(config.MODEL.WIDTH, config.MODEL.HEIGHT))

            im_in = random_jpg_quality(im_in)
            im_in = random_bright(im_in)

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
                    string_print = "Epoch = %d g_step = %d Current_Loss = %.6f " % (epoch,g_step, current)
                    utils.LOG(string_print)
                    summary_writer.add_summary(summary_str, global_step=g_step)

                if g_step % config.TRAIN.save_step == 0:
                    saver.save(sess, checkpoint_path + "/model.ckpt", global_step=global_step)
                    
                input_image_batch = []
                output_image_batch = []
            else:
                continue

        epoch_time=time.time()-epoch_st
        remain_time=epoch_time*(config.TRAIN.num_epochs-1-epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        if s!=0:
            train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
        else:
            train_time="Remaining training time : Training completed.\n"
        utils.LOG(train_time)
