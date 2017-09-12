from collections import defaultdict
import numpy as np
import tensorflow as tf
import random
from model import DocumentModel

def load_data(fname, N, D):
    x = defaultdict(list)
    with open(fname) as f:
        for line in f:
            if len(x["doc_len"])>=N:
                break
            x["doc_len"].append(0)
            data_line = []
            for s in line.split(','):
                k,v = [int(s1) for s1 in s.split(':')]
                if k < D:
                    data_line.append([k,v])

            # sort the entry to satisfy tensorflow sparsetensor requirement
            data_line = sorted(data_line, key=lambda x: x[0])
            #total=sum([v for k, v in data_line])
            for k, v in data_line:
                x["indices"].append(k)
                x["data"].append(v)
                x["data2"].append(1)
            x["doc_len"][-1]+=len(data_line)

    x["indices"] = np.array(x["indices"])
    x["data"] = np.array(x["data"])
    x["data2"] = np.array(x["data2"])
    x["doc_len"] = np.array(x["doc_len"])
    x["doc_len_offset"] = np.cumsum(x["doc_len"]) - x["doc_len"]

    return {"x": x}

def get_doc_by_idx(x_data, x_idx, D):
    x = x_data["x"]
    doc_count = len(x_idx)
    indices_idx = np.concatenate([x["doc_len_offset"][ix] + np.arange(x["doc_len"][ix]) for ix in x_idx])
    indices_num = np.repeat(np.arange(doc_count), x["doc_len"][x_idx])
    indices = np.stack([indices_num, x["indices"][indices_idx]], axis=1)
    data = x["data"][indices_idx]
    data2 = x["data2"][indices_idx]
    return indices, data, data2, [doc_count, D]

def generate_batch(x_idx, x_data, n_iter, batch_size, D):
    random.seed(123)
    batch_mul = len(x_idx) * 1.0 / batch_size
    for n in range(1, n_iter):
        batch_id = (n - 1) % int(batch_mul)
        if batch_id == 0:
            random.shuffle(x_idx)
        # sort the batch id
        batch_idx = sorted(x_idx[batch_id * batch_size: batch_id * batch_size + batch_size])
        M = len(x_idx) * 1.0 / len(batch_idx)
        batch_data, batch_data2 = get_sparsetensorvalue(batch_idx, x_data, D)
        yield n, batch_data, batch_data2, M

def get_sparsetensorvalue(x_idx, x_data, D):
    x_indices, x_values, x2_values, x_shape = get_doc_by_idx(x_data, x_idx, D)
    x_stv = tf.SparseTensorValue(indices=x_indices, values=x_values, dense_shape=x_shape)
    x2_stv = tf.SparseTensorValue(indices=x_indices, values=x2_values, dense_shape=x_shape)
    return x_stv, x2_stv

def main(N_doc = 10000, D = 2000, batch_size = 1000, max_iter = 100001, keep_prob = 0.75, learning_rate=0.01):
    model_spec = {"K0": 50, "K1": 15, "D": D, "B": 4, "sigma": 0.1, "H0": 300, "H1": 300, "H2": 300}
    x_data = load_data("yelp100000.txt", N_doc, D)
    N_train = N_doc - 1000
    x_train_idx = list(range(N_train))

    N_valid = N_doc  - N_train
    x_valid_idx = list(list(range(N_train, N_train + N_valid)))
    valid_data, valid_data2 = get_sparsetensorvalue(x_valid_idx, x_data, D)

    ELBO_R = np.zeros(max_iter)
    with tf.Session() as sess:
        model = DocumentModel(sess, model_spec, learning_rate)
        if N_valid > 0:
            ELBO_R[0], _, _ = model.valid(valid_data)

        for n, train_data, train_data2, M in generate_batch(x_train_idx, x_data, max_iter, batch_size, D):
            #print("iter:", n, end='\r')
            model.train(train_data, keep_prob, M)
            if N_valid > 0:
                ELBO_R[n], Z_params, W_params = model.valid(valid_data)
                converge = (ELBO_R[n] - ELBO_R[n - 1]) / abs(ELBO_R[n - 1])
                print("iter: %d" % n, "converge: %0.6f" % converge, "val-elbo: %0.5f" % ELBO_R[n])
                if n % 100==0:
                    W0_alpha, W0_mean, W1_alpha, W1_mean = W_params
                    Z0_alpha, Z0_mean, Z1_alpha, Z1_mean = Z_params
                    W0 = np.stack([W0_alpha, W0_mean], axis=0)
                    W1 = np.stack([W1_alpha, W1_mean], axis=0)
                    Z0 = np.stack([Z0_alpha, Z0_mean], axis=0)
                    Z1 = np.stack([Z1_alpha, Z1_mean], axis=0)
                    filename = ('results/Yelp_'+ '%05d' % n + '_W0'+ '.npy')
                    np.save(filename, W0)
                    filename = ('results/Yelp_'+ '%05d' % n + '_W1'+ '.npy')
                    np.save(filename, W1)
                    filename = ('results/Yelp_'+ '%05d' % n + '_Z0'+ '.npy')
                    np.save(filename, Z0)
                    filename = ('results/Yelp_'+ '%05d' % n + '_Z1'+ '.npy')
                    np.save(filename, Z1)

if __name__ == '__main__':
    main()