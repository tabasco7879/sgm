import numpy as np
import numpy.random as npr
import tensorflow as tf
from tensorflow.contrib.distributions import Gamma
from tensorflow.contrib.distributions import Poisson
from .gamma_ars import gamma_ars
from .gamma_ars import gradient_gamma_ars

tf.set_random_seed(123)
npr.seed(123)

z_shp = 0.1
z_rte = 0.1
min_z_alpha = 1e-3
min_z_mean = 1e-4

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
    def __init__(self, sess, model_spec, learning_rate=0.002):
        np.random.seed(123)

        self.model_spec = model_spec
        self.learning_rate = learning_rate

        # tf Graph input
        self.x = tf.sparse_placeholder(tf.float32, [None, self.model_spec["D"]], name="x")
        self.keep_prob = tf.placeholder(tf.float32)

        # Create autoencoder network
        self._create_network()

        self._create_optimizer()

        # Initializing the tensor flow variables
        self.init = tf.global_variables_initializer()

        # Launch the session
        self.sess = sess
        self.sess.run(self.init)

    def _create_network(self):
        K0 = self.model_spec["K0"]
        K1 = self.model_spec["K1"]
        D = self.model_spec["D"]
        B = self.model_spec["B"]
        sigma = self.model_spec["sigma"]

        z0_alpha, z0_mean, z1_alpha, z1_mean = self._inference_network()
        self.z = [z0_alpha, z0_mean, z1_alpha, z1_mean]

        z0, h_val_z0, h_der_z0 = gamma_ars(z0_alpha, z0_mean, B)
        z1, h_val_z1, h_der_z1 = gamma_ars(z1_alpha, z1_mean, B)

        # introduce W0 and W1
        self.log_p_mean = self._generator_network(z0, z1)

        [g_logp_z0, g_logp_z1, g_W0, g_W1] = tf.gradients(self.log_p_mean, [z0, z1, self.W0, self.W1])
        g_logp_z0_alpha, g_logp_z0_mean = gradient_gamma_ars(g_logp_z0, z0_alpha, z0_mean, h_val_z0, h_der_z0)
        g_logp_z1_alpha, g_logp_z1_mean = gradient_gamma_ars(g_logp_z1, z1_alpha, z1_mean, h_val_z1, h_der_z1)

        g_logp_z0_alpha = tf.stop_gradient(g_logp_z0_alpha)
        g_logp_z0_mean = tf.stop_gradient(g_logp_z0_mean)
        g_logp_z1_alpha = tf.stop_gradient(g_logp_z1_alpha)
        g_logp_z1_mean = tf.stop_gradient(g_logp_z1_mean)
        g_W0 = tf.stop_gradient(g_W0)
        g_W1 = tf.stop_gradient(g_W1)

        h_z0_mean = tf.reduce_mean(tf.reduce_sum(entropy(z0_alpha, z0_mean), axis=1))
        h_z1_mean = tf.reduce_mean(tf.reduce_sum(entropy(z1_alpha, z1_mean), axis=1))
        h_z_mean = tf.add(h_z0_mean, h_z1_mean)

        self.elbo = h_z_mean + self.log_p_mean

        proxy_obj_z0 = tf.add(tf.reduce_sum(tf.multiply(g_logp_z0_alpha, z0_alpha)), tf.reduce_sum(tf.multiply(g_logp_z0_mean, z0_mean)))
        proxy_obj_z1 = tf.add(tf.reduce_sum(tf.multiply(g_logp_z1_alpha, z1_alpha)), tf.reduce_sum(tf.multiply(g_logp_z1_mean, z1_mean)))
        proxy_obj_z = tf.add(proxy_obj_z0, proxy_obj_z1)
        proxy_obj_W = tf.reduce_sum(tf.multiply(g_W0, self.W0)) + tf.reduce_sum(tf.multiply(g_W1, self.W1))
        self.proxy_obj = tf.add(proxy_obj_z, proxy_obj_W)

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

            layer_1 = tf.nn.softplus(tf.add(tf.sparse_tensor_dense_matmul(self.x, self.H1), self.H1_bias))
            layer_2 = tf.tanh(tf.add(tf.matmul(layer_1, self.H0), self.H0_bias))
            layer_do = tf.nn.dropout(layer_2, self.keep_prob)

            z0_alpha = tf.add(tf.nn.softplus(tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, self.H0_z0_alpha), self.H0_z0_alpha_bias))), min_z_alpha)
            z0_mean = tf.add(tf.nn.softplus(tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, self.H0_z0_mean), self.H0_z0_mean_bias))), min_z_mean)

            z1_alpha = tf.add(tf.nn.softplus(tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, self.H0_z1_alpha), self.H0_z1_alpha_bias))), min_z_alpha)
            z1_mean = tf.add(tf.nn.softplus(tf.contrib.layers.batch_norm(tf.add(tf.matmul(layer_do, self.H0_z1_mean), self.H0_z1_mean_bias))), min_z_mean)

            return (z0_alpha, z0_mean, z1_alpha, z1_mean)

    def _generator_network(self, z0, z1):
        with tf.variable_scope("generator"):
            K0 = self.model_spec["K0"]
            K1 = self.model_spec["K1"]
            D = self.model_spec["D"]
            sigma = self.model_spec['sigma']

            self.W0 = tf.get_variable("W0", dtype=tf.float32, initializer=tf.random_normal([K0, D], stddev=sigma, dtype=tf.float32))
            self.W1 = tf.get_variable("W1", dtype=tf.float32, initializer=tf.random_normal([K1, K0], stddev=sigma, dtype=tf.float32))
            gamma_z1 = Gamma(tf.to_float(z_shp), tf.to_float(z_rte))
            log_prior_z1 = tf.reduce_sum(gamma_z1.log_prob(z1), axis=1)

            z0_rte = tf.div(tf.to_float(z_shp), tf.nn.softplus(tf.matmul(z1, self.W1)))
            gamma_z0 = Gamma(tf.to_float(z_shp), z0_rte)
            log_prior_z0 = tf.reduce_sum(gamma_z0.log_prob(z0), axis=1)

            poisson_rate = tf.nn.softplus(tf.matmul(z0, self.W0))
            poisson_x = Poisson(poisson_rate)
            log_likelihood = tf.reduce_sum(poisson_x.log_prob(tf.sparse_tensor_to_dense(self.x)), axis=1)

            log_p_mean = tf.reduce_mean(tf.add(tf.add(log_prior_z1, log_prior_z0), log_likelihood))
            return log_p_mean

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.99).minimize(-self.proxy_obj)

    def train(self, X, keep_prob):
        return self.sess.run((self.optimizer), feed_dict={self.x: X, self.keep_prob: keep_prob})

    def valid(self, X):
        return self.sess.run((self.elbo), feed_dict={self.x: X, self.keep_prob: 1.0})
