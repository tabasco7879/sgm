import tensorflow as tf

def gamma_h(epsilon, alpha):
    """
    Reparameterization for gamma rejection sampler without shape augmentation.
    """
    b = alpha - 1. / 3.
    c = 1. / (9. * b) ** 0.5
    v = 1. + epsilon * c

    return b * (v ** 3)

def gamma_grad_h(epsilon, alpha):
    """
    Gradient of reparameterization without shape augmentation.
    """
    b = alpha - 1. / 3.
    c = 1. / (9. * b) ** 0.5
    v = 1. + epsilon * c

    return v ** 3 - 13.5 * epsilon * b * (v ** 2) * (c ** 3)

def gamma_h_boosted(epsilon, u, alpha):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    B = tf.shape(u)[1]
    K = tf.shape(alpha)[0]
    alpha1 = tf.expand_dims(alpha, 0)
    B1 = tf.expand_dims(tf.to_float(tf.range(B)), 0)
    alpha_vec = tf.transpose(tf.tile(alpha1,[B,1])) + tf.tile(B1,[K,1])
    u_pow = tf.pow(u,1. / alpha_vec)

    return tf.reduce_prod(u_pow,axis=1) * gamma_h(epsilon, alpha + tf.to_float(B))

def gamma_grad_h_boosted(epsilon, u, alpha):
    """
    Gradient of reparameterization with shape augmentation.
    """
    B = tf.shape(u)[1]
    K = tf.shape(alpha)[0]
    h_val = gamma_h(epsilon, alpha + tf.to_float(B))
    h_der = gamma_grad_h(epsilon, alpha + tf.to_float(B))
    alpha1 = tf.expand_dims(alpha, 0)
    B1 = tf.expand_dims(tf.to_float(tf.range(B)), 0)
    alpha_vec = tf.transpose(tf.tile(alpha1,[B,1])) + tf.tile(B1,[K,1])
    u_pow = tf.pow(u, 1. / alpha_vec)
    u_der = -tf.log(u) / alpha_vec ** 2.

    return tf.reduce_prod(u_pow, axis=1) * h_val * (h_der / h_val + tf.reduce_sum(u_der,axis=1))

def gamma_grad_logr(epsilon, alpha):
    """
    Gradient of log-proposal.
    """
    b = alpha - 1. / 3.
    c = 1. / (9. * b) ** 0.5
    v = 1. + epsilon * c

    return -0.5 / b + 9. * epsilon * (c ** 3) / v

def gamma_grad_logq(epsilon, alpha):
    """
    Gradient of log-Gamma at proposed value.
    """
    h_val = gamma_h(epsilon, alpha)
    h_der = gamma_grad_h(epsilon, alpha)

    return tf.log(h_val) + (alpha - 1.) * h_der / h_val - h_der - tf.polygamma(tf.to_float(0), alpha)

def gamma_correction(epsilon, alpha):
    """
    Correction term grad (log q - log r)
    """
    return gamma_grad_logq(epsilon, alpha) - gamma_grad_logr(epsilon,alpha)

def calc_epsilon(p, alpha):
    """
    Calculate the epsilon accepted by Numpy's internal Marsaglia & Tsang
    rejection sampler. (h is invertible)
    """
    sqrtAlpha = (9. * alpha - 3.) ** 0.5
    powZA = (p / (alpha - 1. / 3.)) ** (1. / 3.)

    return sqrtAlpha * (powZA - 1.)

def gamma_ars(alpha, m, B):
    alpha0 = tf.reshape(alpha, [-1])
    m0 = tf.reshape(m, [-1])
    zw0, h_val, h_der = gamma_ars0(alpha0, m0, B)
    zw = tf.reshape(zw0, tf.shape(alpha))
    return zw, h_val, h_der

def gamma_ars0(alpha, m, B):
    num_z = tf.shape(alpha)[0]

    # generate the shape-augmented gamma samples
    lmbda = tf.random_gamma(shape=[], alpha=alpha + tf.to_float(B), beta=1., dtype=tf.float32)
    lmbda = tf.clip_by_value(lmbda, 1e-5, 1e+30)

    # generate the uniform samples
    u = tf.random_uniform([num_z, B], dtype=tf.float32)
    u = tf.clip_by_value(u, 1e-30, 1.)

    # compute epsilon
    epsilon = calc_epsilon(lmbda, alpha + tf.to_float(B))

    h_val = gamma_h_boosted(epsilon, u, alpha)
    h_der = gamma_grad_h_boosted(epsilon, u, alpha)

    # compute the final gamma samples
    zw = h_val * m / alpha
    zw = tf.clip_by_value(zw, 1e-5, 1e+30)
    return zw, h_val, h_der

def gradient_gamma_ars(gradient_zw, alpha, m, h_val, h_der):
    gradient_zw0 = tf.reshape(gradient_zw, [-1])
    alpha0 = tf.reshape(alpha, [-1])
    m0 = tf.reshape(m, [-1])
    gradient_alpha0, gradient_mean0 = gradient_gamma_ars0(gradient_zw0, alpha0, m0, h_val, h_der)
    gradient_alpha = tf.reshape(gradient_alpha0, tf.shape(alpha))
    gradient_mean = tf.reshape(gradient_mean0, tf.shape(m))
    return gradient_alpha, gradient_mean

def gradient_gamma_ars0(gradient_zw, alpha, m, h_val, h_der):
    gradient_alpha = gradient_zw * m * (alpha * h_der - h_val) / alpha ** 2
    gradient_mean = gradient_zw * h_val / alpha
    return gradient_alpha, gradient_mean