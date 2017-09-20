import tensorflow as tf

class SGOptimizer(object):
    def __init__(self, func_obj, var_list, eta):
        self.func_obj = func_obj
        self.eta = eta
        self.var_list = var_list
        self._iter = tf.Variable(0., trainable=False)
        self.var_hist_list = []
        # Create slots for the first and second moments.
        for i, v in enumerate(self.var_list):
            v_hist = tf.get_variable(name = 'hist_1' + str(i), initializer = tf.zeros(tf.shape(v)), dtype=tf.float32)
            self.var_hist_list.append(v_hist)
        gradients = tf.gradients(self.func_obj, self.var_list)
        self.v1_list=[]
        for g, v, v_hist in zip(gradients, self.var_list, self.var_hist_list):
            v_hist1 = tf.assign(v_hist, 0.1 * tf.pow(g, 2.0) + 0.9 * v_hist)
            iter1 = tf.assign(self._iter, self._iter + 1.)
            step = self.eta * tf.pow(iter1, -0.5 + 1e-16)/(1. + tf.sqrt(v_hist1))
            v1 = tf.assign(v, v + step*g)
            self.v1_list.append(v1)

    def maximize(self):
        return self.v1_list

