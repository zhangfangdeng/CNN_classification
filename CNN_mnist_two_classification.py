#CNN,mnist二分类,Le-Net
#判断手写数字是0或非0
#用到的模板为手写数字样本数据库mnist_uint8.mat，其中保存了6000个训练样本和10000个测试样本。
#其中： train_x为一个60000*784的矩阵即样本图像的分辨率转换成线性的数据表示；train_y为一个60000*10的矩阵其中10代表数字一共有10个分类。
#要对数据进行二分类，就判断为0或非0，只需提取train_y的第一列。
#将train_y中第一列放到一个两列的矩阵trainy中，trainy的第二列的值根据第一列确定，如果第一列的值为1，则第二列的值为0，如此就改成了两类的问题，即将train_x中所有的样本都改成两类。

#训练1000步，到950步的时候准确率达到0.98

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_DATA/",one_hot=True)

BATCH_SIZE = 100
LEARNING_RATE = 1e-4

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

#输入测试值，计算准确度
def compute_accuracy(v_xs,v_ys):
  global y_conv
  #进行预测，得到y_pre
  y_pre = sess.run(y_conv,feed_dict={xs:v_xs,keep_prob:1.0})
  correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
  result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
  return result

# 初始化所有的权值 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,2])

#第一层卷积
x_image = tf.reshape(xs,[-1,28,28,1]) #输入层 28*28*1
W_conv1 = weight_variable([5,5,1,32])#[5,5,1,32]分别代表5*5*1的kernel，输入层有1个feature maps，输出层共32个feature maps
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)

#第一层池化
h_pool1 = max_pool(h_conv1)

#第二层卷积
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)

#第二层池化
h_pool2 = max_pool(h_conv2)

#第三层 全连接
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#第四层 softmax输出层
W_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#模型评估
cross_entropy = -tf.reduce_mean(ys * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_y = mnist.train.next_batch(BATCH_SIZE)
        batch_ys = trainy(batch_y)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys,keep_prob:1.0})

        # 每50步 使用mnist.test.images对模型进行测试
        if i % 50 == 0:
            batch_xs, batch_y = mnist.test.next_batch(BATCH_SIZE)
            batch_ys = trainy(batch_y)
            print(i, compute_accuracy(batch_xs, batch_ys))
    # 保存模型
    save_path = saver.save(sess, 'save/save_CNNFunctionNet.ckpt')

