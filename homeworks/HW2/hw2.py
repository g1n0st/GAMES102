import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameter
training_epochs = 200010
learning_rate = 0.01
sampling_points = 200
w0 = 0.0

train_X = np.asarray([1.1, 2.2, 3.3, 3.7, 4.4, 5.5, 5.7, 5.8, 6.0, 7.997])
train_Y = np.asarray([3.2, 1.9, 1.7, 2.5, 2.76, 3.366, 0.3, 8.9, 4.2, 2.596])

w0 = np.average(train_Y) # choose w0 as average of Y

n_samples = train_X.shape[0]
X = tf.placeholder('float')
Y = tf.placeholder('float')

w = tf.Variable(tf.random_normal(shape = [sampling_points], mean = 0, stddev = 1), name = "w")
b = tf.Variable(tf.random_normal(shape = [sampling_points], mean = 0, stddev = 1), name = "b")
v = tf.Variable(tf.random_normal(shape = [sampling_points], mean = 0, stddev = 1), name = "v") 

base_funcs = tf.multiply(tf.exp(tf.multiply(tf.pow(tf.add(tf.multiply(X, w), b), 2), -0.5)), v) # f_i(x) = v * exp(-(wx+b)^2/2)
cost = tf.pow(tf.reduce_sum(base_funcs) + w0 - Y, 2) / (2 * n_samples) # L2 loss function

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        if epoch % 100 == 0: print ('now training epoch is: {}'.format(epoch))
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict = {X: x, Y: y})

        # draw points in (px, py)
        if epoch in {100, 500, 1000, 5000, 10000, 50000, 100000, 20000}:
            points_num = 2500
            px = np.linspace(0.0, 10.0, points_num, endpoint = True, retstep = False, dtype = float)
            py = (np.sum(sess.run(v) * np.exp(-0.5 * (np.multiply(sess.run(w), px.reshape(points_num, 1)) + sess.run(b)) ** 2), axis = 1) + w0)
            plt.plot(px, py, label = 'NN, epoch = {}'.format(epoch))


    plt.plot(train_X, train_Y, 'ro', label = 'Original data')
    plt.legend()
    plt.show()
    

