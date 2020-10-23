import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

def Guass(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2) / np.sqrt(2 * np.pi) / sigma
# Hyperparameter
training_epochs = 2000
learning_rate = 0.01
funcs_num = 50
w0 = 0.0
w0_strategy = 'average'

# random data with some stiff error terms
train_X = np.asarray([1.1, 2.2, 3.3, 3.7, 4.4, 5.5, 5.7, 5.8, 6.0, 7.997])
train_Y = np.asarray([3.2, 1.9, 1.7, 2.5, 2.76, 3.366, 0.3, 8.9, 4.2, 2.596])
# data with random noise
'''
train_X = np.asarray([ 0.        ,  0.52631579,  1.05263158,  1.57894737,  2.10526316,
        2.63157895,  3.15789474,  3.68421053,  4.21052632,  4.73684211,
        5.26315789,  5.78947368,  6.31578947,  6.84210526,  7.36842105,
        7.89473684,  8.42105263,  8.94736842,  9.47368421, 10.        ])
train_Y = np.asarray([25.12093998, 21.2635335 , 14.62626735, 13.23894537,  6.1715793 ,
        3.53320113,  4.56597062,  2.82891303, -1.61361493,  0.27643671,
        1.60571712, -0.86539428,  2.4306359 ,  3.5382783 ,  5.88626305,
        8.329499  , 14.04783885, 14.05547953, 18.5881014 , 24.22668917])
'''
'''
# data without noise
train_X = np.asarray([ 0.        ,  0.52631579,  1.05263158,  1.57894737,  2.10526316,
        2.63157895,  3.15789474,  3.68421053,  4.21052632,  4.73684211,
        5.26315789,  5.78947368,  6.31578947,  6.84210526,  7.36842105,
        7.89473684,  8.42105263,  8.94736842,  9.47368421, 10.        ])
train_Y = np.asarray([25.        , 20.01385042, 15.58171745, 11.70360111,  8.37950139,
        5.60941828,  3.3933518 ,  1.73130194,  0.6232687 ,  0.06925208,
        0.06925208,  0.6232687 ,  1.73130194,  3.3933518 ,  5.60941828,
        8.37950139, 11.70360111, 15.58171745, 20.01385042, 25.        ])
'''
n_samples = train_X.shape[0]

#for funcs_num in {50, 100, 10000, 50000}:
#for w0_strategy in {'average', 'zero', 'minimum'}:
if w0_strategy == 'average': w0 = np.average(train_Y) # choose w0 as average of Y
if w0_strategy == 'zero': w0 = 0.0
if w0_strategy == 'minimum': w0 = np.min(train_Y)

X = tf.placeholder('float')
Y = tf.placeholder('float')

w = tf.Variable(tf.random_normal(shape = [funcs_num], mean = 0, stddev = 1), name = "w")
b = tf.Variable(tf.random_normal(shape = [funcs_num], mean = 0, stddev = 1), name = "b")
v = tf.Variable(tf.random_normal(shape = [funcs_num], mean = 0, stddev = 1), name = "v") 

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
    points_num = 2500
    px = np.linspace(0.0, 10.0, points_num, endpoint = True, retstep = False, dtype = float)
    py = (np.sum(sess.run(v) * np.exp(-0.5 * (np.multiply(sess.run(w), px.reshape(points_num, 1)) + sess.run(b)) ** 2), axis = 1) + w0)
    plt.plot(px, py, label = 'NN, func_num = {}, epoch = {}, strategy = {}'.format(funcs_num, training_epochs, w0_strategy))

    # sigma = 1.0
    for sigma in {0.05, 0.1, 0.15}:
        A = Guass(train_X - train_X.reshape(train_X.shape[0], 1), sigma)
        inv_A = np.linalg.inv(A)
        base = inv_A @ (train_Y - w0).reshape(train_Y.shape[0], 1)
        guass_y = np.sum(Guass(px - train_X.reshape(train_X.shape[0], 1), sigma) * base, axis = 0) + w0
        plt.plot(px, guass_y, label = 'Guass, sigma = {}'.format(sigma))

plt.plot(train_X, train_Y, 'ro', label = 'Original data')
plt.legend()
plt.show()
    

