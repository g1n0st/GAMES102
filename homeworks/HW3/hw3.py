import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

def Gauss(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2) / np.sqrt(2 * np.pi) / sigma

# Hyperparameter
training_epochs = 10
learning_rate = 0.01
funcs_num = 10
w0_x, w0_y = 0.0, 0.0
w0_strategy = 'average'
parameterization_strategy = 'uniform'

# random 1D data with some stiff error terms
'''
train_X = np.asarray([1.1, 2.2, 3.3, 3.7, 4.4, 5.5, 5.7, 5.8, 6.0, 7.997])
train_Y = np.asarray([3.2, 1.9, 1.7, 2.5, 2.76, 3.366, 0.3, 8.9, 4.2, 2.596])
'''

# Helix function
train_X = np.asarray([ 0.00000000e+00, -1.41341910e+00, -3.22389330e+01,  1.25942300e+01,
        6.15769744e+01, -3.42872554e+01, -8.52576632e+01,  6.51852508e+01,
        1.00807362e+02, -1.03401028e+02, -1.06161809e+02,  1.46552575e+02,
        9.97754767e+01, -1.91871194e+02, -8.07090481e+01,  2.36328326e+02]) * 0.1
train_Y = np.asarray([    0.        ,   -16.30247961,     5.6325466 ,    47.44789486,
         -22.19400241,   -74.28727342,    48.69086448,    94.18887529,
         -83.51685945,  -104.86888789,   124.52521733,   104.51001316,
        -169.12578733,   -91.86042312,   214.40311141,    66.30879605]) * 0.1

# Piecewise continuous linear function
'''
train_X = np.asarray([0.1 , 0.2 , 0.5 , 0.9 , 1.  , 2.  , 3.3 , 3.5 , 3.55, 4.2 , 4.7 ,
       4.75, 6.2 , 8.8 , 8.94, 9.5 ])

train_Y = np.asarray([-0.1 , -0.2 , -0.5 , -0.9 , -1.  , -2.  , -3.3 , -3.5 , -3.55,
       -4.2 , -4.7 , -4.75, -3.8 , -1.2 , -1.06, -0.5 ])
'''

n_samples = train_X.shape[0]

t = np.zeros(n_samples) # parameterization of samples

def uniform_parameterization():
    for i in range(1, n_samples): t[i] = t[i - 1] + 1.0

def chordal_parameterization():
    for i in range(1, n_samples): 
        t[i] = t[i - 1] + np.linalg.norm(np.asarray([train_X[i] - train_X[i - 1], train_Y[i] - train_Y[i - 1]]))

def centripetal_parameterization():
    for i in range(1, n_samples): 
        t[i] = t[i - 1] + np.sqrt(np.linalg.norm(np.asarray([train_X[i] - train_X[i - 1], train_Y[i] - train_Y[i - 1]])))

def foley_parameterization():
    for i in range(1, n_samples):
        k1 = np.asarray([train_X[i - 1], train_Y[i - 1]])
        k2 = np.asarray([train_X[i], train_Y[i]])
        k0 = (k1 if i == 1 else np.asarray([train_X[i - 2], train_Y[i - 2]]))
        k3 = (k2 if i == n_samples - 1 else np.asarray([train_X[i + 1], train_Y[i + 1]]))
        k01 = k1 - k0
        k12 = k2 - k1
        k23 = k3 - k2
        n01 = np.linalg.norm(k01) + 1e-5 # ||ki - ki-1||
        n12 = np.linalg.norm(k12) + 1e-5 # ||ki+1 - ki||
        n23 = np.linalg.norm(k23) + 1e-5 # ||ki+2 - ki+1||
        a_i = np.arccos(k01.dot(k12) / (n01 * n12)) # ai = angle(ki-1, ki, ki+1)
        a_i1 = np.arccos(k12.dot(k23) / (n12 * n23))
        ahat_i = min(np.pi - a_i, np.pi / 2)
        ahat_i1 = min(np.pi - a_i1, np.pi / 2)
        t[i] = t[i - 1] + n12 * (1 + 3 / 2 * ahat_i * n01 / (n01 + n12) + 3 / 2 * ahat_i1 * n12 / (n12 + n23))

for (parameterization_strategy, sigma) in zip(['uniform', 'chordal', 'centripetal'], [1, 10, 5]):
    # init w0
    if w0_strategy == 'average': w0_x, w0_y = np.average(train_X), np.average(train_Y)
    if w0_strategy == 'zero': w0_x, w0_y = 0.0, 0.0
    if w0_strategy == 'minimum': w0_x, w0_y = np.min(train_X), np.min(train_Y)

    # parameterization
    if parameterization_strategy == 'uniform': uniform_parameterization()
    if parameterization_strategy == 'chordal': chordal_parameterization()
    if parameterization_strategy == 'centripetal': centripetal_parameterization()
    if parameterization_strategy == 'foley': foley_parameterization()

    X = tf.placeholder('float')
    Y = tf.placeholder('float')
    T = tf.placeholder('float')

    # neural network for x-coordinate
    wx = tf.Variable(tf.random_normal(shape = [funcs_num], mean = 0, stddev = 1), name = "wx")
    bx = tf.Variable(tf.random_normal(shape = [funcs_num], mean = 0, stddev = 1), name = "bx")
    vx = tf.Variable(tf.random_normal(shape = [funcs_num], mean = 0, stddev = 1), name = "vx")

    # neural network for y-coordinate
    wy = tf.Variable(tf.random_normal(shape = [funcs_num], mean = 0, stddev = 1), name = "wy")
    by = tf.Variable(tf.random_normal(shape = [funcs_num], mean = 0, stddev = 1), name = "by")
    vy = tf.Variable(tf.random_normal(shape = [funcs_num], mean = 0, stddev = 1), name = "vy")

    base_x = tf.multiply(tf.exp(tf.multiply(tf.pow(tf.add(tf.multiply(T, wx), bx), 2), -0.5)), vx) # fx_i(t) = vx*exp(-(wx*t+bx)^2/2)
    base_y = tf.multiply(tf.exp(tf.multiply(tf.pow(tf.add(tf.multiply(T, wy), by), 2), -0.5)), vy) # fy_i(t) = vy*exp(-(wy*t+bx)^2/2)
    cost = (tf.pow(tf.reduce_sum(base_x) + w0_x - X, 2) + tf.pow(tf.reduce_sum(base_y) + w0_y - Y, 2)) / (4 * n_samples) # L2 loss function

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            if epoch % 100 == 0: print ('now training epoch is: {}'.format(epoch))
            for (x, y, t0) in zip(train_X, train_Y, t):
                sess.run(optimizer, feed_dict = {X: x, Y: y, T : t0})

        # draw points in (px, py)
        points_num = 2500
        pt = np.linspace(t[0], t[n_samples - 1], points_num, endpoint = True, retstep = False, dtype = float)
        px = (np.sum(sess.run(vx) * np.exp(-0.5 * (np.multiply(sess.run(wx), pt.reshape(points_num, 1)) + sess.run(bx)) ** 2), axis = 1) + w0_x)
        py = (np.sum(sess.run(vy) * np.exp(-0.5 * (np.multiply(sess.run(wy), pt.reshape(points_num, 1)) + sess.run(by)) ** 2), axis = 1) + w0_y)
        # plt.plot(px, py, label = 'NN, epoch = {}, parameterization = {}, func = {}'.format(training_epochs, parameterization_strategy, funcs_num))

        A = Gauss(t - t.reshape(n_samples, 1), sigma)
        inv_A = np.linalg.inv(A)
        base_x = inv_A @ (train_X - w0_x).reshape(train_X.shape[0], 1)
        gauss_x = np.sum(Gauss(pt - t.reshape(n_samples, 1), sigma) * base_x, axis = 0) + w0_x
        base_y = inv_A @ (train_Y - w0_y).reshape(train_Y.shape[0], 1)
        gauss_y = np.sum(Gauss(pt - t.reshape(n_samples, 1), sigma) * base_y, axis = 0) + w0_y
        plt.plot(gauss_x, gauss_y, label = 'Gauss, sigma = {}, parameterization = {}, w0 = {}'.format(sigma, parameterization_strategy, w0_strategy))

plt.plot(train_X, train_Y, 'ro', label = 'Original data')
plt.legend() 
plt.show()