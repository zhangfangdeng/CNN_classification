#全连接网络mnist二分类
#判断手写数字是0或非0
#用到的模板为手写数字样本数据库mnist_uint8.mat，其中保存了6000个训练样本和10000个测试样本。
# 其中： train_x为一个60000*784的矩阵即样本图像的分辨率转换成线性的数据表示；train_y为一个60000*10的矩阵其中10代表数字一共有10个分类。
#要对数据进行二分类，就判断为0或非0，只需提取train_y的第一列。
#将train_y中第一列放到一个两列的矩阵trainy中，trainy的第二列的值根据第一列确定，如果第一列的值为1，则第二列的值为0，如此就改成了两类的问题，即将train_x中所有的样本都改成两类。

#训练1000步，到950步的时候准确率达到0.92

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)

LEARNING_RATE = 0.1
BATCH_SIZE = 100

#计算准确度
def compute_accuracy(v_xs,v_ys):
  global prediction
  #进行预测，得到y_pre
  y_pre = sess.run(prediction,feed_dict={xs:v_xs})
  correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
  return result

#将十分类矩阵改为二分类矩阵
def trainy(y):
    trainy = tf.zeros(BATCH_SIZE, 2)
    trainy = y[:,0:2]
    for i in range(BATCH_SIZE):
        if trainy[i,0] == 1:
            trainy[i,1] = 0
        else:
            trainy[i,1] = 1
    return trainy

#构建全连接层
def add_layer(inputs,in_size,out_size,activation_function = None):
  Weight = tf.Variable(tf.random_normal([in_size,out_size]))
  biases = tf.Variable(tf.zeros([1,out_size])+0.1)
  Wx_plus_b = tf.matmul(inputs,Weight) + biases
  if activation_function is None:
    outputs = Wx_plus_b
  else:
    outputs = activation_function(Wx_plus_b)
  return outputs

#定义占位符
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,2])

#添加隐藏层 784个输入节点，10 个输出节点
l1 = add_layer(xs,784,10,activation_function=tf.nn.tanh)
#添加输出层 10个输入节点，2个输出节点
prediction = add_layer(l1,10,2,activation_function=tf.nn.tanh)
#定义损失函数和反向传播算法
#使用交叉熵作为损失函数
#tf.clip_by_value(t, clip_value_min, clip_value_max,name=None)
#基于min和max对张量t进行截断操作，为了应对梯度爆发或者梯度消失的情况
cross_entropy = -tf.reduce_mean(ys * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

#定义初始化和保存
init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)
  for i in range(1000):
    batch_xs,batch_y = mnist.train.next_batch(BATCH_SIZE)
    batch_ys = trainy(batch_y)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})

    # 每50步 使用mnist.test.images对模型进行测试
    if i%50==0:
      batch_xs, batch_y = mnist.test.next_batch(BATCH_SIZE)
      batch_ys = trainy(batch_y)
      print(i,compute_accuracy(batch_xs,batch_ys))

  #保存模型
  save_path = saver.save(sess, 'save/save_FunctionNet.ckpt')

