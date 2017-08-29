import numpy as np
import numpy.random as npr
import tensorflow as tf
from tensorflow.contrib.distributions import Gamma
from tensorflow.contrib.distributions import Poisson
from .gamma_ars import gamma_ars
from .gamma_ars import gradient_gamma_ars

tf.set_random_seed(123)
npr.seed(123)

Z_shp = 0.1
Z_rte = 0.1
W_shp = 0.1
W_rte = 0.3
min_Alpha = 1e-3
min_Mean = 1e-4

def entropy(alpha, m):
    gamma_z = Gamma(alpha, alpha / m)
    entropy = gamma_z.entropy()
    return entropy

def xavier_init(fan_in, fan_out, constant=0.1):
    """ Xavier initialization of network weights"""
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

#tf.reset_default_graph()

class DocumentModel(object):
    def __init__(self, sess, model_spec, learning_rate=0.01):
        np.random.seed(123)

        self.model_spec = model_spec
        self.learning_rate = learning_rate

        # tf Graph input
        self.x = tf.sparse_placeholder(tf.float32, [None, self.model_spec["D"]], name="x")
        self.keep_prob = tf.placeholder(tf.float32)
        self.M = tf.placeholder(tf.float32)

        # Create autoencoder network
        self._create_network()

        self._create_optimizer()

        # Initializing the tensor flow variables
        self.init = tf.global_variables_initializer()

        # Launch the session
        self.sess = sess
        self.sess.run(self.init)

    def _create_network(self):
        B = self.model_spec["B"]

        W0_alpha, W0_mean, W1_alpha, W1_mean = self._W_network()
        Z0_alpha, Z0_mean, Z1_alpha, Z1_mean = self._inference_network()
        self.z_param = [Z0_alpha, Z0_mean, Z1_alpha, Z1_mean]

        z0, h_val_z0, h_der_z0 = gamma_ars(Z0_alpha, Z0_mean, B)
        z1, h_val_z1, h_der_z1 = gamma_ars(Z1_alpha, Z1_mean, B)
        w0, h_val_w0, h_der_w0 = gamma_ars(W0_alpha, W0_mean, B)
        w1, h_val_w1, h_der_w1 = gamma_ars(W1_alpha, W1_mean, B)

        self.z = [z0, z1]

        self.log_p, self.log_prior_w, self.log_p_data = self._generator_network(z0, z1, w0, w1)

        [g_logp_z0, g_logp_z1, g_logp_w0, g_logp_w1] = tf.gradients(self.log_p, [z0, z1, w0, w1])
        g_logp_Z0_alpha, g_logp_Z0_mean = gradient_gamma_ars(g_logp_z0, Z0_alpha, Z0_mean, h_val_z0, h_der_z0)
        g_logp_Z1_alpha, g_logp_Z1_mean = gradient_gamma_ars(g_logp_z1, Z1_alpha, Z1_mean, h_val_z1, h_der_z1)
        g_logp_W0_alpha, g_logp_W0_mean = gradient_gamma_ars(g_logp_w0, W0_alpha, W0_mean, h_val_w0, h_der_w0)
        g_logp_W1_alpha, g_logp_W1_mean = gradient_gamma_ars(g_logp_w1, W1_alpha, W1_mean, h_val_w1, h_der_w1)

        g_logp_Z0_alpha = tf.stop_gradient(g_logp_Z0_alpha)
        g_logp_Z0_mean = tf.stop_gradient(g_logp_Z0_mean)
        g_logp_Z1_alpha = tf.stop_gradient(g_logp_Z1_alpha)
        g_logp_Z1_mean = tf.stop_gradient(g_logp_Z1_mean)
        g_logp_W0_alpha = tf.stop_gradient(g_logp_W0_alpha)
        g_logp_W0_mean = tf.stop_gradient(g_logp_W0_mean)
        g_logp_W1_alpha = tf.stop_gradient(g_logp_W1_alpha)
        g_logp_W1_mean = tf.stop_gradient(g_logp_W1_mean)

        h_Z0 = tf.reduce_sum(entropy(Z0_alpha, Z0_mean))
        h_Z1 = tf.reduce_sum(entropy(Z1_alpha, Z1_mean))
        h_W0 = tf.reduce_sum(entropy(W0_alpha, W0_mean))
        h_W1 = tf.reduce_sum(entropy(W1_alpha, W1_mean))
        self.h_Z = tf.add(h_Z0, h_Z1)
        self.h_W = tf.add(h_W0, h_W1)

        self.elbo = tf.div(tf.add(self.h_Z, self.log_p_data), tf.to_float(tf.shape(self.x)[0]))

        proxy_obj_Z0 = tf.add(tf.reduce_sum(tf.multiply(g_logp_Z0_alpha, Z0_alpha)), tf.reduce_sum(tf.multiply(g_logp_Z0_mean, Z0_mean)))
        proxy_obj_Z1 = tf.add(tf.reduce_sum(tf.multiply(g_logp_Z1_alpha, Z1_alpha)), tf.reduce_sum(tf.multiply(g_logp_Z1_mean, Z1_mean)))
        proxy_obj_Z = tf.add(tf.add(proxy_obj_Z0, proxy_obj_Z1), tf.multiply(self.h_Z, self.M))

        proxy_obj_W0 = tf.add(tf.reduce_sum(tf.multiply(g_logp_W0_alpha, W0_alpha)), tf.reduce_sum(tf.multiply(g_logp_W0_mean, W0_mean)))
        proxy_obj_W1 = tf.add(tf.reduce_sum(tf.multiply(g_logp_W1_alpha, W1_alpha)), tf.reduce_sum(tf.multiply(g_logp_W1_mean, W1_mean)))
        proxy_obj_W = tf.add(tf.add(proxy_obj_W0, proxy_obj_W1), self.h_W)

        self.proxy_obj = tf.add(proxy_obj_Z, proxy_obj_W)

    def _W_network(self):
        with tf.variable_scope("W"):
            K0 = self.model_spec["K0"]
            K1 = self.model_spec["K1"]
            D = self.model_spec["D"]
            sigma = self.model_spec["sigma"]

            self.log_W0_alpha = tf.get_variable("log_W0_alpha", dtype=tf.float32, initializer=tf.random_normal([K0, D], mean=0.5, stddev=sigma, dtype=tf.float32))
            self.log_W0_mean = tf.get_variable("log_W0_mean", dtype=tf.float32, initializer=tf.random_normal([K0, D], stddev=sigma, dtype=tf.float32))
            self.log_W1_alpha = tf.get_variable("log_W1_alpha", dtype=tf.float32, initializer=tf.random_normal([K1, K0], mean=0.5, stddev=sigma, dtype=tf.float32))
            self.log_W1_mean = tf.get_variable("log_W1_mean", dtype=tf.float32, initializer=tf.random_normal([K1, K0], stddev=sigma, dtype=tf.float32))

            W0_alpha = tf.add(tf.nn.softplus(self.log_W0_alpha), min_Alpha)
            W0_mean = tf.add(tf.nn.softplus(self.log_W0_mean), min_Mean)
            W1_alpha = tf.add(tf.nn.softplus(self.log_W1_alpha), min_Alpha)
            W1_mean = tf.add(tf.nn.softplus(self.log_W1_mean), min_Mean)

            return (W0_alpha, W0_mean, W1_alpha, W1_mean)

    def _inference_network(self):
        with tf.variable_scope("inference"):
            K0 = self.model_spec["K0"]
            K1 = self.model_spec["K1"]
            H0 = self.model_spec["H0"]
            H1 = self.model_spec["H1"]
            D = self.model_spec["D"]

            self.H0_z0_alpha = tf.get_variable("H0_z0_alpha", dtype=tf.float32, initializer=xavier_init(H0, K0))
            self.H0_z0_alpha_bias = tf.get_variable("H0_z0_alpha_bias", dtype=tf.float32, initializer=tf.zeros([K0], dtype=tf.float32))
            self.H0_z0_mean = tf.get_variable("H0_z0_mean", dtype=tf.float32, initializer=xavier_init(H0, K0))
            self.H0_z0_mean_bias = tf.get_variable("H0_z0_mean_bias", dtype=tf.float32, initializer=tf.zeros([K0], dtype=tf.float32))

            self.H0_z1_alpha = tf.get_variable("H0_z1_alpha", dtype=tf.float32, initializer=xavier_init(H0, K1))
            self.H0_z1_alpha_bias = tf.get_variable("H0_z1_alpha_bias", dtype=tf.float32, initializer=tf.zeros([K1], dtype=tf.float32))
            self.H0_z1_mean = tf.get_variable("H0_z1_mean", dtype=tf.float32, initializer=xavier_init(H0, K1))
            self.H0_z1_mean_bias = tf.get_variable("H0_z1_mean_bias", dtype=tf.float32, initializer=tf.zeros([K1], dtype=tf.float32))

            self.H0 = tf.get_variable("H0", dtype=tf.float32, initializer=xavier_init(H1, H0))
            self.H0_bias = tf.get_variable("H0_bias", dtype=tf.float32, initializer=tf.zeros([H0], dtype=tf.float32))
            self.H1 = tf.get_variable("H1", dtype=tf.float32, initializer=xavier_init(D, H1))
            self.H1_bias = tf.get_variable("H1_bias", dtype=tf.float32, initializer=tf.zeros([H1], dtype=tf.float32))

            layer_1 = tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(self.x, self.H1), self.H1_bias))
            layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, self.H0), self.H0_bias))
            layer_do = tf.nn.dropout(layer_2, self.keep_prob)

            self.check_value = tf.reduce_max(self.H1)
            self.check_value1 = tf.reduce_max(self.H1_bias)
            self.check_value2 = tf.reduce_min(self.H1)
            self.check_value3 = tf.reduce_min(self.H1_bias)

            Z0_alpha = tf.add(tf.nn.softplus(tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, self.H0_z0_alpha), self.H0_z0_alpha_bias))), min_Alpha)
            Z0_mean = tf.add(tf.nn.softplus(tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, self.H0_z0_mean), self.H0_z0_mean_bias))), min_Mean)

            Z1_alpha = tf.add(tf.nn.softplus(tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, self.H0_z1_alpha), self.H0_z1_alpha_bias))), min_Alpha)
            Z1_mean = tf.add(tf.nn.softplus(tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, self.H0_z1_mean), self.H0_z1_mean_bias))), min_Mean)

            return (Z0_alpha, Z0_mean, Z1_alpha, Z1_mean)

    def _generator_network(self, z0, z1, w0, w1):
        K0 = self.model_spec["K0"]
        K1 = self.model_spec["K1"]
        D = self.model_spec["D"]

        gamma_W = Gamma(tf.to_float(W_shp), tf.to_float(W_rte))
        log_prior_w0 = tf.reduce_sum(gamma_W.log_prob(w0))
        log_prior_w1 = tf.reduce_sum(gamma_W.log_prob(w1))

        gamma_Z1 = Gamma(tf.to_float(Z_shp), tf.to_float(Z_rte))
        log_prior_z1 = tf.reduce_sum(gamma_Z1.log_prob(z1))

        Z0_mean = tf.matmul(z1, w1)
        Z0_rte = tf.div(tf.to_float(Z_shp), Z0_mean)
        gamma_Z0 = Gamma(tf.to_float(Z_shp), Z0_rte)
        log_prior_z0 = tf.reduce_sum(gamma_Z0.log_prob(z0))

        poisson_rate = tf.matmul(z0, w0)
        poisson_X = Poisson(poisson_rate)
        log_likelihood = tf.reduce_sum(poisson_X.log_prob(tf.sparse_tensor_to_dense(self.x)))

        log_prior_w = tf.add(log_prior_w0, log_prior_w1)
        log_p_data = tf.add(tf.add(log_prior_z1, log_prior_z0), log_likelihood)
        log_p = tf.add(tf.multiply(log_p_data, self.M), log_prior_w)
        return log_p, log_prior_w, log_p_data

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.99).minimize(-self.proxy_obj)

    def train(self, X, keep_prob, M):
        return self.sess.run((self.optimizer), feed_dict={self.x: X, self.keep_prob: keep_prob, self.M: M})

    def valid(self, X):
        return self.sess.run((self.elbo), feed_dict={self.x: X, self.keep_prob: 1.0, self.M: 1.0})
