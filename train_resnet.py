#! /usr/bin/python
# -*- coding: utf8 -*-

import sys
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
import math
from PIL import Image
import os
import io
import resnet_model

_BATCH_NORM_DECAY = 0.99
_BATCH_NORM_EPSILON = 1e-3
_WEIGHT_DECAY = 1e-4
resnet_num = 110
n_epoch = 160
print_freq = 1
model_file_name = "model_cifar10_tfrecord.ckpt"
batch_size = 100
resume = False# load model, resume from previous checkpoint?
log_dir = './log'
# ratio = tf.placeholder(tf.float32, [], name='ratio')

train_data = "/data/cifar10Old/train.cifar10"
test_data = "/data/cifar10Old/test.cifar10"

train_length = 0
test_length = 0
for _ in tf.python_io.tf_record_iterator(train_data):
  train_length +=1
for _ in tf.python_io.tf_record_iterator(test_data):
  test_length +=1

print("train data lenght is: %f", train_length)
print("test data lenght is: %f", test_length)
n_step_epoch = int(train_length/batch_size)
n_step = n_epoch * n_step_epoch


def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
    if is_train == True:
        # 1. Padding to size 40x40 accroding to paper resnet
        img = tf.image.resize_image_with_crop_or_pad(img, 40, 40)
        # 2. Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [32, 32, 3])
        # 3. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)
        # 4. Randomly change brightness.
        img = tf.image.per_image_standardization(img)

    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        # img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        img = tf.image.per_image_standardization(img)

    elif is_train == None:
        img = img

    label = tf.cast(features['label'], tf.int32)
    return img, label

with tf.device('/cpu:0'):

    tl.files.exists_or_mkdir(log_dir)

    # prepare data in cpu
    x_train_, y_train_ = read_and_decode(train_data, True)
    x_test_, y_test_   = read_and_decode(test_data, False)

    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
        batch_size=batch_size, capacity=2000, min_after_dequeue=1000, num_threads=16) # set the number of threads here
    # for testing, uses batch instead of shuffle_batch
    x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
        batch_size=batch_size, capacity=50000, num_threads=16)


    with tf.device('/gpu'): # <-- remove it if you don't have GPU
        y_ = y_train_batch
        network = resnet_model.cifar10_resnet_v2_generator(resnet_num, 10)
        y_0 = network(x_train_batch,'resnet', True, False)
        ce_0 = tl.cost.cross_entropy(y_0, y_, name='cost')
        cost = ce_0# + L2
        correct_prediction = tf.equal(tf.cast(tf.argmax(y_0, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        learning_rate = tf.Variable(0.01, trainable=False)
        # lr_decay = tf.assign(learning_rate, learning_rate * 0.1)

        #for testing
        y_ = y_test_batch
        network_t = resnet_model.cifar10_resnet_v2_generator(resnet_num, 10)
        y_0_t = network_t(x_test_batch, 'resnet', False, True)
        cost_test = tl.cost.cross_entropy(y_0_t, y_, name='cost')
        correct_prediction = tf.equal(tf.cast(tf.argmax(y_0_t, 1), tf.int32), y_)
        acc_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    ## train
    with tf.device('/gpu'):   # <-- remove it if you don't have GPU
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_op = tf.train.AdamOptimizer(learning_rate, use_locking=False).minimize(cost)
            train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_locking=False, use_nesterov=True).minimize(cost + l2_loss * _WEIGHT_DECAY)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    saver = tf.train.Saver(max_to_keep=1)
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    summaries.add(tf.summary.scalar('ratio', resnet_model.ratio))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    change_step = int(n_step/2)
    step = 0; best_acc_0 = 0; best_acc_1 = 0;
    for epoch in range(n_epoch):
        sys.stdout.flush()
        start_time = time.time()
        train_loss, train_acc, n_batch = 0, 0, 0
        if epoch == 0:
            lr_assign_op = tf.assign(learning_rate, learning_rate * 1.0)
            sess.run(lr_assign_op)
        if epoch == 1:
            lr_assign_op = tf.assign(learning_rate, learning_rate * 10.0)
            sess.run(lr_assign_op)
        if epoch == 80 or epoch == 120:
            lr_assign_op = tf.assign(learning_rate, learning_rate * 0.1)
            sess.run(lr_assign_op)
        for s in range(n_step_epoch):
            rate = float(change_step-step)/change_step
            rate = max(0.0, rate)
            err, ac, _ = sess.run([cost, acc, train_op], feed_dict={resnet_model.ratio: rate})
            # print('ratio=', sess.run(resnet_model.ratio, feed_dict={resnet_model.ratio: rate}))
            step += 1;
            train_loss += err;
            train_acc += ac;
            n_batch += 1

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d : Step %d-%d of %d took %fs" % (
                epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            # write train accuracy and loss
            summary1 = tf.Summary(value=[
                tf.Summary.Value(tag="train_loss", simple_value=train_loss / n_batch),
                tf.Summary.Value(tag="train_acc", simple_value=train_acc / n_batch),
            ])
            train_acc_list.append(train_acc / n_batch)
            train_loss_list.append(train_loss / n_batch)
            summary_writer.add_summary(summary1, epoch+1)
            summary_writer.flush()
            summary_str = sess.run(summary_op, feed_dict={resnet_model.ratio: rate})
            summary_writer.add_summary(summary_str, epoch+1)
            summary_writer.flush()

            #test
            test_loss, test_acc, test_batch = 0, 0, 0
            for _ in range(int(test_length / batch_size)):
                err, ac = sess.run([cost_test, acc_test], feed_dict={resnet_model.ratio: rate})
                test_loss += err;
                test_acc += ac;
                test_batch += 1
            t1 = "   test loss: %f" % (test_loss / test_batch)
            t2 = "   test acc: %f" % (test_acc / test_batch)
            print(t1)
            print(t2)
            test_acc_list.append(test_acc / test_batch)
            test_loss_list.append(test_loss / test_batch)
            summary2 = tf.Summary(value=[
                tf.Summary.Value(tag="test_loss", simple_value=test_loss / test_batch),
                tf.Summary.Value(tag="test_acc", simple_value=test_acc / test_batch),
            ])
            summary_writer.add_summary(summary2, epoch+1)
            summary_writer.flush()

        if(epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(log_dir, model_file_name)
            saver.save(sess,checkpoint_path,global_step=epoch+1)

    # write file
    str1 = 'train_acc_list=%s' % (train_acc_list)
    str2 = 'train_loss_list=%s' % (train_loss_list)
    str3 = 'test_acc_list=%s' % (test_acc_list)
    str4 = 'test_loss_list=%s' % (test_loss_list)
    file = open('accuracylist.txt', 'a')
    file.write(str1 + '\r\n')  # ubuntu aught to '\r\n'
    file.write(str2 + '\r\n')
    file.write(str3 + '\r\n')
    file.write(str4 + '\r\n')
    file.close()
    coord.request_stop()
    coord.join(threads)
    summary_writer.close()
    sess.close()
