import numpy as np
import time
from pyDOE import lhs
import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import pickle
import math
import scipy.io

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CPU:-1; GPU0: 1; GPU1: 0;
np.random.seed(1111)
tf.set_random_seed(1111)

class DeepHPM:
    # Initialize the class
    def __init__(self, Collo, SRC, IC, ABS, MSE, uv_layers, E_layers, lb, ub, uvDir='', eDir=''):

        # Count for callback function
        self.count = 0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties, 1D case
        self.rho = 1.0

        # P wave velocity: sqrt((lam+2nu)/rho)=1.732

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.t_c = Collo[:, 1:2]

        # Source wave
        self.x_SRC = SRC[:, 0:1]
        self.t_SRC = SRC[:, 1:2]
        self.s22_SRC = SRC[:, 2:3]

        # Initial condition point, t=0
        self.x_IC = IC[:, 0:1]
        self.t_IC = IC[:, 1:2]

        # Absorption boundary condition
        self.x_ABS = ABS[:, 0:1]
        self.t_ABS = ABS[:, 1:2]

        # Measurement
        self.x_MSE = MSE[:, 0:1]
        self.t_MSE = MSE[:, 1:2]
        self.u_MSE = MSE[:, 2:3]

        # Define layers
        self.uv_layers = uv_layers
        self.E_layers = E_layers

        # Initialize NNs
        if uvDir == '':
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        if eDir == '':
            self.E_weights, self.E_biases = self.initialize_NN(self.E_layers)
        else:
            self.E_weights, self.E_biases = self.load_NN(eDir, self.E_layers)


        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float64, shape=[])
        self.x_tf = tf.placeholder(tf.float64, shape=[None, self.x_c.shape[1]])    # Point for postprocessing
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t_c.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float64, shape=[None, self.x_c.shape[1]])
        self.t_c_tf = tf.placeholder(tf.float64, shape=[None, self.t_c.shape[1]])

        self.x_SRC_tf = tf.placeholder(tf.float64, shape=[None, self.x_SRC.shape[1]])
        self.t_SRC_tf = tf.placeholder(tf.float64, shape=[None, self.t_SRC.shape[1]])
        # self.u_SRC_tf = tf.placeholder(tf.float64, shape=[None, self.u_SRC.shape[1]])
        self.s22_SRC_tf = tf.placeholder(tf.float64, shape=[None, self.s22_SRC.shape[1]])

        self.x_MSE_tf = tf.placeholder(tf.float64, shape=[None, self.x_MSE.shape[1]])
        self.t_MSE_tf = tf.placeholder(tf.float64, shape=[None, self.t_MSE.shape[1]])
        self.u_MSE_tf = tf.placeholder(tf.float64, shape=[None, self.u_MSE.shape[1]])

        self.x_IC_tf = tf.placeholder(tf.float64, shape=[None, self.x_IC.shape[1]])
        self.t_IC_tf = tf.placeholder(tf.float64, shape=[None, self.t_IC.shape[1]])

        self.x_ABS_tf = tf.placeholder(tf.float64, shape=[None, self.x_ABS.shape[1]])
        self.t_ABS_tf = tf.placeholder(tf.float64, shape=[None, self.t_ABS.shape[1]])

        # tf graphs
        self.u_pred, self.s_pred = self.net_us(self.x_tf, self.t_tf)
        self.e_pred = self.net_e(self.x_tf, self.t_tf)
        self.E_pred = self.net_E(self.x_tf)

        self.u_MSE_pred, _ = self.net_us(self.x_MSE_tf, self.t_MSE_tf)

        self.u_IC_pred, _ = self.net_us(self.x_IC_tf, self.t_IC_tf)
        self.ut_IC_pred = self.net_ut(self.x_IC_tf, self.t_IC_tf)
        _, self.s_SRC_pred = self.net_us(self.x_SRC_tf, self.t_SRC_tf)

        self.ux_ABS_pred = self.net_e(self.x_ABS_tf, self.t_ABS_tf)
        self.ut_ABS_pred = self.net_ut(self.x_ABS_tf, self.t_ABS_tf)

        self.f_pred_u, self.f_pred_s = self.net_f_sig(self.x_c_tf, self.t_c_tf)

        # self.loss_f_uv = tf.reduce_mean(tf.abs(self.f_pred_u))
        # self.loss_f_s = tf.reduce_mean(tf.abs(self.f_pred_s))
        # self.loss_IC = tf.reduce_mean(tf.abs(self.u_IC_pred)) \
        #                  + tf.reduce_mean(tf.abs(self.ut_IC_pred))
        # self.loss_SRC = tf.reduce_mean(tf.abs(self.s_SRC_pred - self.s22_SRC_tf))
        # self.loss_MSE = tf.reduce_mean(tf.abs(self.u_MSE_pred - self.u_MSE_tf))
        # self.loss_ABS = tf.reduce_mean(tf.abs(6.5**0.5*self.ux_ABS_pred + self.ut_ABS_pred))

        self.loss_f_uv = tf.reduce_mean(tf.square(self.f_pred_u))
        self.loss_f_s = tf.reduce_mean(tf.square(self.f_pred_s))
        self.loss_IC = tf.reduce_mean(tf.square(self.u_IC_pred)) \
                         + tf.reduce_mean(tf.square(self.ut_IC_pred))
        self.loss_SRC = tf.reduce_mean(tf.square(self.s_SRC_pred - self.s22_SRC_tf))
        self.loss_MSE = tf.reduce_mean(tf.square(self.u_MSE_pred - self.u_MSE_tf))
        self.loss_ABS = tf.reduce_mean(tf.square(6.5**0.5*self.ux_ABS_pred + self.ut_ABS_pred))

        self.loss = 3*(self.loss_f_uv + self.loss_f_s) + self.loss_IC + self.loss_SRC + 3*self.loss_ABS + 50*self.loss_MSE

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(100*self.loss,
                                                                var_list=self.uv_weights + self.uv_biases + self.E_weights + self.E_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 300000,
                                                                         'maxfun': 300000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases + self.E_weights + self.E_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)

    def save_NN_uv(self, fileDir):
        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)
        with open(fileDir, 'wb') as f:
            # pickle.dump([np.array(uv_weights), np.array(uv_biases)], f)
            pickle.dump([uv_weights, uv_biases], f)
            print("Save NN_uv parameters successfully...")

    def save_NN_E(self, fileDir):
        E_weights = self.sess.run(self.E_weights)
        E_biases = self.sess.run(self.E_biases)
        with open(fileDir, 'wb') as f:
            # pickle.dump([np.array(uv_weights), np.array(uv_biases)], f)
            pickle.dump([E_weights, E_biases], f)
            print("Save NN_E parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)
            # print(len(uv_weights))
            # print(np.shape(uv_weights))
            # print(num_layers)

            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num], dtype=tf.float64)
                b = tf.Variable(uv_biases[num], dtype=tf.float64)
                weights.append(W)
                biases.append(b)
                print("Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_E(self, X, weights, biases): # Multiply the X by a constant
        num_layers = len(weights) + 1
        H = X
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_us(self, x, t):
        # This NN return sigma_phi
        us = self.neural_net(tf.concat([x, t], 1), self.uv_weights, self.uv_biases)
        u = us[:, 0:1]
        s = us[:, 1:2]
        return u, s

    def net_E(self, x):
        # This NN return sigma_phi
        E = self.neural_net_E(x, self.E_weights, self.E_biases)
        return E

    def net_ut(self, x, t):
        u, _ = self.net_us(x, t)
        # Strains
        ut = tf.gradients(u, t)[0]
        return ut

    def net_e(self, x, t):
        u, _ = self.net_us(x, t)
        # Strains
        e = tf.gradients(u, x)[0]
        return e

    def net_f_sig(self, x, t):

        # E = 1.5 + 4.5 / (1.0 + tf.math.exp(-200.0 * (x - 1.0)))
        E = self.net_E(x)

        # E = 1.5
        # fig = plt.figure()
        # ax = fig.gca()
        # X = np.arange(0, 2.0, 0.01)
        # Z = 1.5 + 4.5 / (1.0 + np.exp(-200.0 * (X - 1.0)))
        # ax.plot(X, Z, marker='o')
        # plt.show()

        rho = self.rho

        u, s = self.net_us(x, t)

        # Strains
        e = self.net_e(x, t)

        # Plane stress problem
        # sp11 = E / (1 - mu * mu) * e11 + E * mu / (1 - mu * mu) * e22
        # sp22 = E * mu / (1 - mu * mu) * e11 + E / (1 - mu * mu) * e22
        # sp12 = E / (2 * (1 + mu)) * e12

        # Plane strain problem
        sp = E * e

        # Cauchy stress
        f_s = s - sp

        s_x = tf.gradients(s, x)[0]
        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]

        # f_u:=Sxx_x+Sxy_y-rho*u_tt
        f_u = s_x - rho*u_tt

        return f_u, f_s

    def callback(self, loss):
        self.count = self.count + 1
        print('{} th iterations, Loss: {}'.format(self.count, loss))

    def train(self, iter, learning_rate, batch_num):

        loss_f_uv = []
        loss_f_s = []
        loss_IC = []
        loss_SRC = []
        loss = []

        # The collocation point is splited into partitions of batch_num，1 epoch for training
        for i in range(batch_num):
            col_num = self.x_c.shape[0]
            idx_start = int(i * col_num / batch_num)
            idx_end = int((i + 1) * col_num / batch_num)

            tf_dict = {self.x_c_tf: self.x_c[idx_start:idx_end,:], self.t_c_tf: self.t_c[idx_start:idx_end,:],
                       self.x_IC_tf: self.x_IC, self.t_IC_tf: self.t_IC,
                       self.x_ABS_tf: self.x_ABS, self.t_ABS_tf: self.t_ABS,
                       self.x_SRC_tf: self.x_SRC, self.t_SRC_tf: self.t_SRC, self.s22_SRC_tf: self.s22_SRC,
                       self.x_MSE_tf: self.x_MSE, self.t_MSE_tf: self.t_MSE, self.u_MSE_tf: self.u_MSE,
                       self.learning_rate: learning_rate}

            for it in range(iter):

                self.sess.run(self.train_op_Adam, tf_dict)

                # Print
                if it % 10 == 0:
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e' %
                          (it, loss_value))

                loss_f_uv.append(self.sess.run(self.loss_f_uv, tf_dict))
                loss_f_s.append(self.sess.run(self.loss_f_s, tf_dict))
                loss_IC.append(self.sess.run(self.loss_IC, tf_dict))
                loss_SRC.append(self.sess.run(self.loss_SRC, tf_dict))
                loss.append(self.sess.run(self.loss, tf_dict))

        return loss_f_uv, loss_f_s, loss_IC, loss_SRC, loss

    def train_bfgs(self, batch_num):
        # The collocation point is splited into partitions of batch_num
        for i in range(batch_num):
            col_num = self.x_c.shape[0]
            idx_start = int(i*col_num/batch_num)
            idx_end = int((i+1)*col_num/batch_num)
            tf_dict = {self.x_c_tf: self.x_c[idx_start:idx_end,:], self.t_c_tf: self.t_c[idx_start:idx_end,:],
                       self.x_IC_tf: self.x_IC, self.t_IC_tf: self.t_IC,
                       self.x_ABS_tf: self.x_ABS, self.t_ABS_tf: self.t_ABS,
                       self.x_SRC_tf: self.x_SRC, self.t_SRC_tf: self.t_SRC, self.s22_SRC_tf: self.s22_SRC,
                       self.x_MSE_tf: self.x_MSE, self.t_MSE_tf: self.t_MSE, self.u_MSE_tf: self.u_MSE}

            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss],
                                    loss_callback=self.callback)

    def predict(self, x_star, t_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.t_tf: t_star})
        s_star = self.sess.run(self.s_pred, {self.x_tf: x_star, self.t_tf: t_star})
        e_star = self.sess.run(self.e_pred, {self.x_tf: x_star, self.t_tf: t_star})
        return u_star, s_star, e_star

    def predict_E(self, x_star):
        E_star = self.sess.run(self.E_pred, {self.x_tf: x_star})
        return E_star


    def getloss(self):  # To be updated

        tf_dict = {self.x_c_tf: self.x_c, self.t_c_tf: self.t_c,
                       self.x_IC_tf: self.x_IC, self.t_IC_tf: self.t_IC,
                       self.x_ABS_tf: self.x_ABS, self.t_ABS_tf: self.t_ABS,
                       self.x_SRC_tf: self.x_SRC, self.t_SRC_tf: self.t_SRC, self.s22_SRC_tf: self.s22_SRC,
                       self.x_MSE_tf: self.x_MSE, self.t_MSE_tf: self.t_MSE, self.u_MSE_tf: self.u_MSE}

        loss_f_uv = self.sess.run(self.loss_f_uv, tf_dict)
        loss_f_s = self.sess.run(self.loss_f_s, tf_dict)
        loss_IC = self.sess.run(self.loss_IC, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)
        loss_SRC = self.sess.run(self.loss_SRC, tf_dict)
        loss_ABS = self.sess.run(self.loss_ABS, tf_dict)
        loss_MSE = self.sess.run(self.loss_MSE, tf_dict)

        print('loss_f_uv:', loss_f_uv)
        print('loss_f_s:', loss_f_s)
        print('loss_IC:', loss_IC)
        print('loss_SRC:', loss_SRC)
        print('loss_ABS:', loss_ABS)
        print('loss_MSE:', loss_MSE)
        print('loss:', loss)


def CartGrid(xmin, xmax, ymin, ymax, tmin, tmax, num, num_t):
    # num: number per edge
    # num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    xx, yy = np.meshgrid(x, y)
    t = np.linspace(tmin, tmax, num=num_t)
    xxx, yyy, ttt = np.meshgrid(x, y, t)
    xxx = xxx.flatten()[:, None]
    yyy = yyy.flatten()[:, None]
    ttt = ttt.flatten()[:, None]
    return xxx, yyy, ttt


def preprocess(dir):
    # dir: directory of training data
    data = scipy.io.loadmat(dir)

    X = data['x']
    U0 = data['u0']
    S11 = data['s11']

    x_star = X.flatten()[:, None]
    u0_star = U0.flatten()[:, None]
    s11_star = S11.flatten()[:, None]

    return x_star, u0_star, s11_star


if __name__ == "__main__":

    PI = math.pi

    MAX_T = 3.5

    # only x, t
    lb = np.array([0.0, 0.0])
    ub = np.array([2.0, MAX_T])

    # Network configuration
    uv_layers = [2] + 4*[80] + [2]
    E_layers = [1] + 2*[10] + [1]


    N_t = int(MAX_T * 20 + 1)  # 4 frames per second, 0.005s/frame in measurement

    # prescribed u1 at t=1.0 on upper boundary
    # u_prscb = 20

    # Initial condition point for u, v
    IC = np.array([0, 0]) + np.array([2.0, 0]) * lhs(2, 1000)

    ABS = np.array([2.0, 0]) + np.array([0, MAX_T]) * lhs(2, 5000)

    # Collocation point
    XYT_c = lb + (ub - lb) * lhs(2, 100000)
    # XYT_c_ext = [0.8, 0] + [0.4, MAX_T] * lhs(2, 10000)
    # XYT_c = np.concatenate((XYT_c, XYT_c_ext), 0)

    # Wave source point in the middle, x=15.0, y=15.0
    t_SRC = np.concatenate((np.linspace(0.0, 0.5, 201), np.linspace(0, MAX_T, 401)))
    t_SRC = t_SRC.flatten()[:, None]
    x_SRC = 0*t_SRC
    x_SRC = x_SRC[1:, :]
    t_SRC = t_SRC[1:, :]
    # v_SRC = Amp*(2*PI**2*(t_SRC-ts)**2/tsh**2-1)*np.exp(-PI**2*(t_SRC-ts)**2/tsh**2)
    s22_SRC = 1.0 * np.exp(-(t_SRC - 0.25) ** 2/0.008)
    SRC = np.concatenate((x_SRC, t_SRC, s22_SRC), 1)

    # u_pred, s_pred, _ = model.predict(x_SRC, t_SRC)

    plt.scatter(t_SRC, s22_SRC, marker='o', alpha=0.2 )
    # plt.scatter(t_SRC, s_pred, marker='o', alpha=0.2)
    plt.show()

    u_MSE = scipy.io.loadmat('u(x=0,t).mat')['u_hist']
    t_MSE = np.linspace(0, 3.5, len(u_MSE))[:, None]
    x_MSE = 0*t_MSE
    MSE = np.concatenate((x_MSE, t_MSE, u_MSE), 1)

    plt.scatter(t_MSE, u_MSE, marker='o', alpha=0.2 )
    plt.show()

    # Visualize ALL the training points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(XYT_c[:,0:1], XYT_c[:,1:2], marker='o', alpha=0.1, s=2, color='red')
    # ax.scatter(SRC[:, 0:1], SRC[:, 1:2], marker='o', alpha=0.2, s=2, color='blue')
    ax.scatter(IC[:, 0:1], IC[:, 1:2], marker='o', alpha=0.2, s=2, color='orange')
    ax.scatter(MSE[:, 0:1], MSE[:, 1:2], marker='o', alpha=0.2, s=2, color='red')
    ax.scatter(ABS[:, 0:1], ABS[:, 1:2], marker='o', alpha=0.2, s=2, color='green')
    ax.set_xlabel('X axis')
    ax.set_ylabel('T axis')
    plt.show()

    with tf.device('/device:GPU:1'):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # model = DeepHPM(XYT_c, SRC, IC, ABS, MSE, uv_layers, E_layers, lb, ub)
        model = DeepHPM(XYT_c, SRC, IC, ABS, MSE, uv_layers, E_layers, lb, ub, uvDir='uv_float64_4x80.pickle', eDir='E_float64_2x10.pickle')

        # # Collocation point
        # model.x_c = XYT_c[:, 0:1]
        # model.t_c = XYT_c[:, 1:2]
        # # Source wave
        # model.x_SRC = SRC[:, 0:1]
        # model.t_SRC = SRC[:, 1:2]
        # model.s22_SRC = SRC[:, 2:3]
        # # Initial condition point, t=0
        # model.x_IC = IC[:, 0:1]
        # model.t_IC = IC[:, 1:2]

        start_time = time.time()
        model.train_bfgs(batch_num=1)
        print("--- %s seconds ---" % (time.time() - start_time))

        model.save_NN_uv('uv_float64_4x80.pickle')
        model.save_NN_E('E_float64_2x10.pickle')

        # with open('uv_float64_2x10.pickle', 'rb') as f:
        #     uv_weights, uv_biases = pickle.load(f)

        model.getloss()

        # shutil.rmtree('./output_nonhomo_abs', ignore_errors=True)
        # os.makedirs('./output_nonhomo_abs')
        # for i in range(N_t):
        #     [x_star, u_star, s11_star] = preprocess(
        #         'C:/Users/raoch/OneDrive/Desktop/Umat/1D-4layer-wave-HF/ProbeData-' + str(10*i) + '.mat')
        #     # Plot the wave height in mid line, y=0 to 28, x=15
        #     time=i*MAX_T/(N_t-1)
        #     x_probe = np.linspace(0, 2.0, 161)
        #     x_probe = x_probe.flatten()[:, None]
        #     t_probe = 0*x_probe + time
        #     u_probe, s_probe, _ = model.predict(x_probe, t_probe)
        #
        #     plt.figure()
        #     plt.scatter(x_star, s11_star, color='blue', label='FEM', s=2, alpha=0.8)
        #     plt.plot(x_probe, s_probe, '--', color='red', label='PINN', alpha=0.8)
        #     plt.text(2.5, 0.5,'T='+str(time))
        #     plt.xlabel('y coordinate')
        #     plt.ylabel('Amplitude')
        #     plt.xlim(0, 2.0)
        #     plt.ylim(-1, 1.8)
        #     plt.legend()
        #     # plt.ylim(-0.25, 0.05)
        #     plt.savefig('./output_nonhomo_abs/'+'stress_'+str(i).zfill(3)+'.png')
        #     plt.close('all')


        ##################### Plot u(x=0,t) w.r.t time #########################
        t_PINN = np.linspace(0, 3.5, 701)
        t_PINN = t_PINN.flatten()[:,None]
        x_PINN = 0*t_PINN
        u_PINN, s_PINN, _ = model.predict(x_PINN, t_PINN)

        u_FEM = scipy.io.loadmat('u(x=0,t).mat')['u_hist']
        t_FEM = np.linspace(0, 3.5, len(u_FEM))[:,None]


        ### Plot the E distribution, In paper

        from matplotlib import rc

        rc('text', usetex=True)
        rc('legend', fontsize=18)

        # Plot the surface displacement prediction
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.subplots_adjust(bottom=0.2)
        ax.plot(t_FEM, u_FEM, color='red', alpha=1, linewidth=1.5, label=r'$\mathrm{Measurements}$')
        ax.plot(t_PINN, u_PINN, '--', color='blue', alpha=1, linewidth=1.5, label=r'$\mathrm{PINN}$')
        ax.set_xlim([0, 3.5])
        ax.set_ylim([-0.16, 0.01])
        ax.set_yticks([-0.15, -0.1, -0.05, 0])
        ax.set_xlabel(r'$\mathrm{Time}$', fontsize=18)
        ax.set_ylabel(r'$\mathrm{Waveform}$', fontsize=18)
        ax.tick_params(axis='y', labelsize=15, direction='in')
        ax.tick_params(axis='x', labelsize=15, direction='in')
        plt.legend()
        plt.savefig('C2_pinn_1D_FWI_surface_disp_4_layers.pdf')
        plt.show()

        # plt.figure()
        # plt.plot(t_FEM, u_FEM, '--', color='blue', alpha=0.8, label='FEM')
        # plt.plot(t_PINN, u_PINN, color='red', label='PINN', alpha=0.8)
        # plt.text(0.5, 0.5, 'T=' + str(time))
        # plt.xlabel('t axis')
        # plt.ylabel('u(x=0)')
        # plt.xlim(0, 3.0)
        # # plt.ylim(-0.25, 0.05)
        # plt.legend()
        # plt.savefig('u(x=0,t).png')
        # plt.show()

        ######################## Visualize E distribution #########################
        plt.close('all')
        x_pred = np.linspace(0, 2.0, 101)
        x_pred = x_pred.flatten()[:,None]
        E_pred = model.predict_E(x_pred)
        E_true = x_pred*0
        dx = 0.1
        for i in range(101):
            if x_pred[i] < 0.5 - dx:
                E_true[i] = 1.5
            elif x_pred[i] < 0.5 + dx:
                E_true[i] = (6.0 + 1.5) / 2 + (6.0 - 1.5) / 2 * np.sin(np.pi * (x_pred[i] - 0.5) / (2 * dx))
            elif x_pred[i] < 1.0 - dx:
                E_true[i] = 6.0
            elif x_pred[i] < 1.0 + dx:
                E_true[i] = (6.0 + 4.0) / 2 + (6.0 - 4.0) / 2 * np.sin(np.pi * (x_pred[i] - 1.0) / (2 * dx) - np.pi)
            elif x_pred[i] < 1.5 - dx:
                E_true[i] = 4.0
            elif x_pred[i] < 1.5 + dx:
                E_true[i] = (4.0 + 6.5) / 2 + (6.5 - 4.0) / 2 * np.sin(np.pi * (x_pred[i] - 1.5) / (2 * dx))
            else:
                E_true[i] = 6.5
        plt.plot(x_pred, E_true, label='FEM', alpha=0.8)
        plt.plot(x_pred, E_pred, '--', label='PINN', alpha=0.8)
        plt.legend()
        plt.show()
        plt.savefig('E(x).png')

        scipy.io.savemat('./E_compa_data_4layers.mat', {'x': x_pred, 'E_PINN': E_pred, 'E_true': E_true,})

        ###################### Plot a diagram as 1D bar ############################
        plt.close('all')
        # cmap = matplotlib.cm.summer
        cmap_reversed = matplotlib.cm.get_cmap('summer_r')
        fig, ax = plt.subplots(figsize=(1.4, 5.0))
        y_pred = np.linspace(0, 2.0, 101)
        x_pred = np.linspace(0, 0.2, 41)
        x_pred, y_pred = np.meshgrid(x_pred, y_pred)
        y_pred = y_pred.flatten()
        x_pred = x_pred.flatten()

        E_true = 0*x_pred
        dx = 0.1
        for i in range(len(y_pred)):
            if y_pred[i] < 0.5 - dx:
                E_true[i] = 1.5
            elif y_pred[i] < 0.5 + dx:
                E_true[i] = (6.0 + 1.5) / 2 + (6.0 - 1.5) / 2 * np.sin(np.pi * (y_pred[i] - 0.5) / (2 * dx))
            elif y_pred[i] < 1.0 - dx:
                E_true[i] = 6.0
            elif y_pred[i] < 1.0 + dx:
                E_true[i] = (6.0 + 4.0) / 2 + (6.0 - 4.0) / 2 * np.sin(np.pi * (y_pred[i] - 1.0) / (2 * dx) - np.pi)
            elif y_pred[i] < 1.5 - dx:
                E_true[i] = 4.0
            elif y_pred[i] < 1.5 + dx:
                E_true[i] = (4.0 + 6.5) / 2 + (6.5 - 4.0) / 2 * np.sin(np.pi * (y_pred[i] - 1.5) / (2 * dx))
            else:
                E_true[i] = 6.5

        E_pred = model.predict_E(y_pred[:, None])
        cf = ax.scatter(x_pred, 2-y_pred, c=E_true, cmap=cmap_reversed, s=4, marker = 's', alpha=1)
        # ax.legend()
        ax.set_xlim([0, 0.2])
        ax.set_ylim([0, 2.0])
        ax.axis('off')
        cbar = fig.colorbar(cf, orientation='horizontal', ax=ax, ticks=[2, 4, 6],fraction=0.046, pad=0.04, aspect=10) #ticks=[3, 4, 5, 6],
        cbar.ax.tick_params(labelsize=18)
        # plt.show()
        plt.savefig('E_true_contour_4layers.png', dpi=300, transparent=True)






        # Plot the inverted E distribution (used in paper)

        from matplotlib import rc

        rc('text', usetex=True)
        rc('legend', fontsize=18)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.subplots_adjust(bottom=0.2)
        ax.plot(x_pred, E_true, color='red', alpha=1, linewidth=1.5, label=r'$\mathrm{Ground\ truth}$')
        ax.plot(x_pred, E_pred, '--', color='blue', alpha=1, linewidth=1.5, label=r'$\mathrm{PINN}$')
        ax.set_xlim([0, 2])
        ax.set_xlabel(r'$\mathrm{Depth}$', fontsize=18)
        ax.set_ylabel(r'$E$', fontsize=18)
        ax.tick_params(axis='y', labelsize=15, direction='in')
        ax.tick_params(axis='x', labelsize=15, direction='in')
        plt.legend()
        plt.savefig('C2_pinn_1D_FWI_E_dist_4_layers.pdf')
        plt.show()

        # Calculate the relative error
        data = scipy.io.loadmat('./E_compa_data_4layers.mat')
        e_pinn = data['E_PINN']
        e_true = data['E_true']

        print("Mean Absolute Relative Error")
        print(np.mean(np.abs(e_pinn - e_true)/np.abs(e_true)))

        pass
