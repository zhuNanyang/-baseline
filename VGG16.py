import tensorflow as tf
import os
import cv2
from Config import config
from model import VGG_model
from read_data import read_data, batch_read_data
from tensorflow.contrib.slim import nets
import numpy as np
con = config()
train_dir = con.train_dir
checkpoint_dir = con.checkpoint_dir
tensorboard_dir = con.tensorboard_dir
tensorboard_train_dir = con.tensorboard_train_dir

if not os.path.isdir(train_dir):os.mkdir(train_dir)
if not os.path.isdir(checkpoint_dir):os.mkdir(checkpoint_dir)
if not os.path.isdir(tensorboard_dir):os.mkdir(tensorboard_dir)
if not os.path.isdir(tensorboard_train_dir):os.mkdir(tensorboard_train_dir)
def train(X, Y, keep_prob):
    #train_image, test_image, train_label, test_label = read_data()
    train_image = np.load('train_image.npy')
    test_image = np.load('test_image.npy')
    train_label = np.load('train_label.npy')
    test_label = np.load('test_label.npy')
    num_batch = int((np.array(train_image).shape[0] - 1) / con.batch_size)
    model = VGG_model(X, Y, keep_prob)

    prediction = model.inference_op()
    print("prediction:{}".format(prediction))
    #cost = tf.reduce_mean(tf.abs((model.Y - prediction) / tf.clip_by_value(tf.abs(model.Y))))
    cost = tf.reduce_mean(tf.reduce_sum(model.Y - prediction))

    '''
    tf.train.Optimizer.minimize(loss, global_step=None, 
    var_list=None, gate_gradients=1, 
    aggregation_method=None, colocate_gradients_with_ops=False, 
    name=None, grad_loss=None)
    添加操作节点，用于最小化loss，并更新var_list 该函数是简单的合并了compute_gradients()与apply_gradients()函数
    返回为一个优化更新后的var_list，如果global_step非None，该操作还会为global_step做自增操作
    
    '''
    optimizer = tf.train.AdamOptimizer(learning_rate=con.learning_rate).minimize(cost)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.reduce_sum(tf.cast(tf.equal(Y, output), tf.int32), axis=1), 4), tf.float32))

    # Summaries
    tf.summary.scalar('train_loss', cost)
    tf.summary.scalar('train_accuracy', accuracy)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    # saver = tf.train.Saver()
    # 将训练好的模型参数保存起来 以便以后进行验证或测试
    # 在创建这个Saver()对象的时候 有一个参数我们经常会用到 就是max_to_keep参数 这个是用来设置保存模型的个数 默认为5 即max_to_keep=5
    # 如果你想每训练一代(epoch)就想保存一次模型 则可以将max_to_keep设置为None或者0
    # 如果你只想保存最后一代的模型 则只需要将max_to_keep设置为1即可










    # 创建为saver对象后 就可以保存训练好的模型了
    # saver.save(sess, 'ckpt/mnist.ckpt', global_step = step)
    # 第一个参数sess 这个就不用说了 第二个参数设定保存的路径和名字第三个参数将训练的次数作为后缀加入到模型名字中
    # saver.save(sess, 'ckpt/mnist.ckpt', global_step = 0) ==> filename: 'ckpt/mnist.ckpt-0'
    # saver.save(sess, 'ckpt/mnsit.ckpt', global_step = 1000) ==> filename: 'ckpt/mnsit.ckpt-1000'
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        train_writer.add_graph(sess.graph)
        # Load the pretrained weights
        #model.load_original_weights(sess, skip_layers=train_layers)
        for epoch in range(con.epoch):
            print("epoch:{}".format(epoch))
            step = 0
            for batch_x, batch_y in batch_read_data(train_image, train_label):
                #print("batch_x:{}".format(batch_x))
                sess.run([optimizer], feed_dict={X:batch_x, Y: batch_y, keep_prob: 0.7})
                if step % con.log_step == 0:
                    summary = sess.run(merged_summary, feed_dict={X:batch_x, Y: batch_y, keep_prob: 0.7})
                    train_writer.add_summary(summary, epoch * num_batch + step)


                step += 1
                if epoch % 100 == 0:
                    #accuracy = sess.run([accuracy], feed_dict={X:batch_x, Y: batch_y, keep_prob: 0.7})
                    #print("accuracy:{}".format(accuracy))
                    # 模型的恢复用restore()函数 它需要两个参数restore(sess, save_path) save_path指的是保存的模型路径
                    # 我们可以使用tf.train.latest_checkpoint()来自动获取最后一次保存的模型的路径
                    # model_file = tf.train.latest_checkpoint('ckpt/')
                    checkpoint_dir_path = os.path.join(checkpoint_dir, 'model_epoch' + str(epoch + 1) + '.ckpt')
                    saver.save(sess, checkpoint_dir_path, global_step=epoch)





if __name__ == "__main__":
    param = config()

    input = tf.placeholder(tf.float32, [None, 256, 256, 3])
    output = tf.placeholder(tf.float32, [None, 4])
    keep_prob = tf.placeholder(tf.float32)
    train(input, output, keep_prob)
