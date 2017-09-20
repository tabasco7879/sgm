import tensorflow as tf
from tensorflow.contrib.distributions import Gamma
from tensorflow.contrib.distributions import Poisson
from .sg_doc import entropy
from .sg_doc import Z_shp
from .sg_doc import Z_rte
from .sg_doc import W_shp
from .sg_doc import W_rte
from .sg_doc import min_Alpha
from .sg_doc import min_Mean
from .gamma_ars import gamma_ars
from .gamma_ars import gradient_gamma_ars
from .sg_optimizer import SGOptimizer

class SegModel(object):
    def __init__(self, sess, model_spec, learning_rate):
        self.model_spec = model_spec
        self.learning_rate = learning_rate

        # tf Graph input
        self.x = tf.sparse_placeholder(tf.float32, [None, self.model_spec["D"]], name="x")
        self.x_idx = tf.placeholder(tf.int64, name="x_idx")

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
        K0 = self.model_spec["K0"]
        K1 = self.model_spec["K1"]

        W0_alpha, W0_mean, W1_alpha, W1_mean = self._W_network()
        Z0_alphas, Z0_means, S0_alphas, S0_means, Z1_alpha, Z1_mean = self._inference_network()
        self.Z_param = [Z0_alphas, Z0_means, S0_alphas, S0_means, Z1_alpha, Z1_mean]
        self.W_param = [W0_alpha, W0_mean, W1_alpha, W1_mean]

        z0_list = []
        h_val_z0_list = []
        h_der_z0_list = []
        for z0_alpha, z0_mean in zip(Z0_alphas, Z0_means):
            z0, h_val_z0, h_der_z0 = gamma_ars(z0_alpha, z0_mean, B)
            z0_list.append(z0)
            h_val_z0_list.append(h_val_z0)
            h_der_z0_list.append(h_der_z0)

        s0_list = []
        h_val_s0_list = []
        h_der_s0_list = []
        for s0_alpha, s0_mean in zip(S0_alphas, S0_means):
            s0, h_val_s0, h_der_s0 = gamma_ars(s0_alpha, s0_mean, B)
            s0_list.append(s0)
            h_val_s0_list.append(h_val_s0)
            h_der_s0_list.append(h_der_s0)

        z1, h_val_z1, h_der_z1 = gamma_ars(Z1_alpha, Z1_mean, B)
        w0, h_val_w0, h_der_w0 = gamma_ars(W0_alpha, W0_mean, B)
        w1, h_val_w1, h_der_w1 = gamma_ars(W1_alpha, W1_mean, B)

        self.logp, self.log_prior_w, self.logp_data = self._generator_network(z0_list, s0_list, z1, w0, w1)

        g_logp_list = tf.gradients(self.logp, z0_list + s0_list + [z1, w0, w1])
        g_logp_z0_list = g_logp_list[:len(z0_list)]
        g_logp_s0_list = g_logp_list[len(z0_list):len(z0_list)+len(s0_list)]
        g_logp_z1, g_logp_w0, g_logp_w1 = g_logp_list[len(z0_list)+len(s0_list):]

        g_logp_Z0_alphas = []
        g_logp_Z0_means = []
        for g_logp_z0, h_val_z0, h_der_z0, z0_alpha, z0_mean in zip(g_logp_z0_list, h_val_z0_list, h_der_z0_list, Z0_alphas, Z0_means):
            g_logp_z0_alpha, g_logp_z0_mean = gradient_gamma_ars(g_logp_z0, z0_alpha, z0_mean, h_val_z0, h_der_z0)
            g_logp_Z0_alphas.append(g_logp_z0_alpha)
            g_logp_Z0_means.append(g_logp_z0_mean)

        g_logp_S0_alphas = []
        g_logp_S0_means = []
        for g_logp_s0, h_val_s0, h_der_s0, s0_alpha, s0_mean in zip(g_logp_s0_list, h_val_s0_list, h_der_s0_list, S0_alphas, S0_means):
            g_logp_s0_alpha, g_logp_s0_mean = gradient_gamma_ars(g_logp_s0, s0_alpha, s0_mean, h_val_s0, h_der_s0)
            g_logp_S0_alphas.append(g_logp_s0_alpha)
            g_logp_S0_means.append(g_logp_s0_mean)

        g_logp_Z1_alpha, g_logp_Z1_mean = gradient_gamma_ars(g_logp_z1, Z1_alpha, Z1_mean, h_val_z1, h_der_z1)
        g_logp_W0_alpha, g_logp_W0_mean = gradient_gamma_ars(g_logp_w0, W0_alpha, W0_mean, h_val_w0, h_der_w0)
        g_logp_W1_alpha, g_logp_W1_mean = gradient_gamma_ars(g_logp_w1, W1_alpha, W1_mean, h_val_w1, h_der_w1)

        g_logp_Z0_alphas = [tf.stop_gradient(g_log_z0_alpha) for g_log_z0_alpha in g_logp_Z0_alphas]
        g_logp_Z0_means = [tf.stop_gradient(g_log_z0_mean) for g_log_z0_mean in g_logp_Z0_means]
        g_logp_S0_alphas = [tf.stop_gradient(g_log_s0_alpha) for g_log_s0_alpha in g_logp_S0_alphas]
        g_logp_S0_means = [tf.stop_gradient(g_log_s0_mean) for g_log_s0_mean in g_logp_S0_means]
        g_logp_Z1_alpha = tf.stop_gradient(g_logp_Z1_alpha)
        g_logp_Z1_mean = tf.stop_gradient(g_logp_Z1_mean)
        g_logp_W0_alpha = tf.stop_gradient(g_logp_W0_alpha)
        g_logp_W0_mean = tf.stop_gradient(g_logp_W0_mean)
        g_logp_W1_alpha = tf.stop_gradient(g_logp_W1_alpha)
        g_logp_W1_mean = tf.stop_gradient(g_logp_W1_mean)

        h_Z0 = 0.
        for z0_alpha, z0_mean in zip(Z0_alphas, Z0_means):
            h_Z0 += tf.reduce_sum(entropy(z0_alpha, z0_mean))

        h_S0 = 0.
        for s0_alpha, s0_mean in zip(S0_alphas, S0_means):
            h_S0 += tf.reduce_sum(entropy(s0_alpha, s0_mean))

        h_Z1 = tf.reduce_sum(entropy(Z1_alpha, Z1_mean))
        h_W0 = tf.reduce_sum(entropy(W0_alpha, W0_mean))
        h_W1 = tf.reduce_sum(entropy(W1_alpha, W1_mean))
        self.h_Z = h_Z0 + h_S0 + h_Z1
        self.h_W = h_W0 + h_W1

        self.elbo = (self.h_Z + self.logp_data) / tf.to_float(tf.shape(self.x)[0])

        proxy_obj_Z0 = 0.
        for z0_alpha, z0_mean, g_logp_z0_alpha, g_logp_z0_mean in zip(Z0_alphas, Z0_means, g_logp_Z0_alphas, g_logp_Z0_means):
            proxy_obj_Z0 += tf.reduce_sum(z0_alpha * g_logp_z0_alpha) + tf.reduce_sum(z0_mean * g_logp_z0_mean)

        proxy_obj_S0 = 0.
        for s0_alpha, s0_mean, g_logp_s0_alpha, g_logp_s0_mean in zip(S0_alphas, S0_means, g_logp_S0_alphas, g_logp_S0_means):
            proxy_obj_S0 += tf.reduce_sum(s0_alpha * g_logp_s0_alpha) + tf.reduce_sum(s0_mean * g_logp_s0_mean)

        proxy_obj_Z1 = tf.reduce_sum(Z1_alpha * g_logp_Z1_alpha) + tf.reduce_sum(Z1_mean * g_logp_Z1_mean)
        self.proxy_obj_Z = proxy_obj_Z0 + proxy_obj_S0 + proxy_obj_Z1 + self.h_Z * self.M

        proxy_obj_W0 = tf.reduce_sum(g_logp_W0_alpha * W0_alpha) + tf.reduce_sum(g_logp_W0_mean * W0_mean)
        proxy_obj_W1 = tf.reduce_sum(g_logp_W1_alpha * W1_alpha) + tf.reduce_sum(g_logp_W1_mean * W1_mean)
        self.proxy_obj_W = proxy_obj_W0 + proxy_obj_W1 + self.h_W

        self.proxy_obj = self.proxy_obj_Z + self.proxy_obj_W

    def _W_network(self):
        with tf.variable_scope("W"):
            K0 = self.model_spec["K0"]
            K1 = self.model_spec["K1"]
            D = self.model_spec["D"]
            sigma = self.model_spec["sigma"]

            self.log_W0_alpha = tf.get_variable("log_W0_alpha", dtype=tf.float32,
                initializer=tf.random_normal([K0, D], mean=0.5, stddev=sigma, dtype=tf.float32))
            self.log_W0_mean = tf.get_variable("log_W0_mean", dtype=tf.float32,
                initializer=tf.random_normal([K0, D], stddev=sigma, dtype=tf.float32))
            self.log_W1_alpha = tf.get_variable("log_W1_alpha", dtype=tf.float32,
                initializer=tf.random_normal([K1, K0], mean=0.5, stddev=sigma, dtype=tf.float32))
            self.log_W1_mean = tf.get_variable("log_W1_mean", dtype=tf.float32,
                initializer=tf.random_normal([K1, K0], stddev=sigma, dtype=tf.float32))

            W0_alpha = tf.add(tf.nn.softplus(self.log_W0_alpha), min_Alpha)
            W0_mean = tf.add(tf.nn.softplus(self.log_W0_mean), min_Mean)
            W1_alpha = tf.add(tf.nn.softplus(self.log_W1_alpha), min_Alpha)
            W1_mean = tf.add(tf.nn.softplus(self.log_W1_mean), min_Mean)

            return (W0_alpha, W0_mean, W1_alpha, W1_mean)

    def _generator_network(self, z0_list, s0_list, z1, w0, w1):
        gamma_W = Gamma(tf.to_float(W_shp), tf.to_float(W_rte))
        log_prior_w0 = tf.reduce_sum(gamma_W.log_prob(w0))
        log_prior_w1 = tf.reduce_sum(gamma_W.log_prob(w1))

        gamma_Z1 = Gamma(tf.to_float(Z_shp), tf.to_float(Z_rte))
        log_prior_z1 = tf.reduce_sum(gamma_Z1.log_prob(z1))

        Z0_mean = tf.matmul(z1, w1)
        Z0_rte = tf.to_float(Z_shp) / Z0_mean
        gamma_Z0 = Gamma(tf.to_float(Z_shp), Z0_rte)
        log_prior_z0 = 0.
        for z0 in z0_list:
            log_prior_z0 += tf.reduce_sum(gamma_Z0.log_prob(z0))

        S0_shp = tf.to_float(W_shp)
        S0_rte = tf.to_float(W_rte)
        log_prior_s0 = 0.
        log_likelihood = 0.

        S = self.model_spec["S"]
        D = self.model_spec["D"]
        x1 = tf.reshape(tf.sparse_tensor_to_dense(self.x), shape = [-1, S, D])
        for n, s0 in enumerate(s0_list):
            gamma_S0 = Gamma(S0_shp, S0_rte)
            S0_rte = S0_shp / s0
            log_prior_s0 += tf.reduce_sum(gamma_S0.log_prob(s0))
            z0_sum = tf.reduce_sum(tf.stack([z0*tf.expand_dims(s0[:,i], axis=1) for i,z0 in enumerate(z0_list)]), axis=0)
            poisson_rate = tf.matmul(z0_sum, w0)
            poisson_X = Poisson(poisson_rate)
            log_likelihood += tf.reduce_sum(poisson_X.log_prob(x1[:,n,:]))

        log_prior_w = log_prior_w0 + log_prior_w1
        logp_data = log_prior_z1 + log_prior_z0 + log_likelihood
        logp = logp_data * self.M + log_prior_w
        return logp, log_prior_w, logp_data

    def _inference_network(self):
        N = self.model_spec["N"]
        T = self.model_spec["T"]
        S = self.model_spec["S"]
        D = self.model_spec["D"]
        sigma = self.model_spec["sigma"]
        K0 = self.model_spec["K0"]
        K1 = self.model_spec["K1"]

        with tf.variable_scope("inference"):
            self.log_Z0_alphas = []
            self.log_Z0_means = []
            for i in range(T):
                log_z0_alpha = tf.get_variable("log_z0_alpha_"+str(i), dtype=tf.float32,
                    initializer=tf.random_normal([N, K0], mean=0.5, stddev=sigma, dtype=tf.float32))
                self.log_Z0_alphas.append(log_z0_alpha)

                log_z0_mean = tf.get_variable("log_z0_mean_"+str(i), dtype=tf.float32,
                    initializer=tf.random_normal([N, K0], stddev=sigma, dtype=tf.float32))
                self.log_Z0_means.append(log_z0_mean)

            self.log_S0_alphas = []
            self.log_S0_means = []
            for i in range(S):
                log_s0_alpha = tf.get_variable("log_s0_alpha_"+str(i), dtype=tf.float32,
                    initializer=tf.random_normal([N, T], mean=0.5, stddev=sigma, dtype=tf.float32))
                self.log_S0_alphas.append(log_s0_alpha)

                log_s0_mean = tf.get_variable("log_s0_mean_"+str(i), dtype=tf.float32,
                    initializer=tf.random_normal([N, T], stddev=sigma, dtype=tf.float32))
                self.log_S0_means.append(log_s0_mean)

            self.log_Z1_alpha = tf.get_variable("log_z1_alpha_"+str(i), dtype=tf.float32,
                initializer=tf.random_normal([N, K1], mean=0.5, stddev=sigma, dtype=tf.float32))

            self.log_Z1_mean = tf.get_variable("log_z1_mean_"+str(i), dtype=tf.float32,
                initializer=tf.random_normal([N, K1], stddev=sigma, dtype=tf.float32))

        Z0_alphas = [tf.nn.softplus(tf.gather(log_z0_alphas, self.x_idx)) for log_z0_alphas in self.log_Z0_alphas]
        Z0_means = [tf.nn.softplus(tf.gather(log_z0_means, self.x_idx)) for log_z0_means in self.log_Z0_means]
        S0_alphas = [tf.nn.softplus(tf.gather(log_s0_alphas, self.x_idx)) for log_s0_alphas in self.log_S0_alphas]
        S0_means = [tf.nn.softplus(tf.gather(log_s0_means, self.x_idx)) for log_s0_means in self.log_S0_means]
        Z1_alpha = tf.nn.softplus(tf.gather(self.log_Z1_alpha, self.x_idx))
        Z1_mean = tf.nn.softplus(tf.gather(self.log_Z1_mean, self.x_idx))
        return Z0_alphas, Z0_means, S0_alphas, S0_means, Z1_alpha, Z1_mean

    def _create_optimizer(self):
        var_list = ([self.log_W0_alpha, self.log_W0_mean, self.log_W1_alpha, self.log_W1_mean,
            self.log_Z1_alpha, self.log_Z1_mean]
            + self.log_Z0_alphas + self.log_Z0_means + self.log_S0_alphas + self.log_S0_means)
        self.optimizer = SGOptimizer(self.proxy_obj, var_list, 0.5).maximize()

    def train(self, X, keep_prob, M, X_idx):
        return self.sess.run((self.optimizer),
            feed_dict={self.x: X, self.keep_prob: keep_prob, self.M: M, self.x_idx: X_idx})

    def valid(self, X, X_idx):
        elbo, z_params, w_params, _ = self.sess.run((self.elbo, self.Z_param, self.W_param, self.optimizer),
            feed_dict={self.x: X, self.keep_prob: 1.0, self.M: 1.0, self.x_idx: X_idx})
        return elbo, z_params, w_params