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
            for k, v in data_line:
                x["indices"].append(k)
                x["data"].append(v)
            x["doc_len"][-1]+=len(data_line)

    x["indices"] = np.array(x["indices"])
    x["data"] = np.array(x["data"])
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
    print(np.sum(data))
    return indices, data, [doc_count, D]

def generate_batch(x_idx, x_data, n_iter, batch_size, D):
    random.seed(123)
    batch_mul = len(x_idx) * 1.0 / batch_size
    for n in range(1, n_iter):
        batch_id = (n - 1) % int(batch_mul)
        if batch_id == 0:
            random.shuffle(x_idx)
        # sort the batch id
        batch_idx = sorted(x_idx[batch_id * batch_size: batch_id * batch_size + batch_size])
        batch_data = get_sparsetensorvalue(batch_idx, x_data, D)
        yield n, batch_data

def get_sparsetensorvalue(x_idx, x_data, D):
    x_indices, x_values, x_shape = get_doc_by_idx(x_data, x_idx, D)
    x_stv = tf.SparseTensorValue(indices=x_indices, values=x_values, dense_shape=x_shape)
    return x_stv

def main(N_doc = 10000, D = 10000, batch_size = 1000, max_iter = 20000, keep_prob = 0.75):
    model_spec = {"K0": 100, "K1": 15, "D": D, "B": 4, "sigma": 0.1, "H0": 100, "H1": 100}
    x_data = load_data("yelp100000.txt", N_doc, D)
    N_train = N_doc - 1000
    x_train_idx = list(range(N_train))

    N_valid = N_doc  - N_train
    x_valid_idx = list(list(range(N_train, N_train + N_valid)))
    valid_data = get_sparsetensorvalue(x_valid_idx, x_data, D)

    ELBO_R = np.zeros(max_iter)
    with tf.Session() as sess:
        model = DocumentModel(sess, model_spec)
        if N_valid > 0:
            ELBO_R[0] = model.valid(valid_data)

        for n, train_data in generate_batch(x_train_idx, x_data, max_iter, batch_size, D):
            #print("iter:", n, end='\r')
            log_p_mean, z_param = sess.run([model.log_p_mean, model.z], feed_dict={model.x: valid_data, model.keep_prob: 1.})
            print(log_p_mean)
            print("z0_alpha", np.max(z_param[0]), np.min(z_param[0]), np.any(np.isnan(z_param[0])))
            print("z0_mean", np.max(z_param[1]), np.min(z_param[1]), np.any(np.isnan(z_param[1])))
            print("z1_alpha", np.max(z_param[2]), np.min(z_param[2]), np.any(np.isnan(z_param[2])))
            print("z1_mean", np.max(z_param[3]), np.min(z_param[3]), np.any(np.isnan(z_param[3])))
            model.train(train_data, keep_prob)
            if N_valid > 0:
                ELBO_R[n] = model.valid(valid_data)
                converge = (ELBO_R[n] - ELBO_R[n - 1]) / abs(ELBO_R[n - 1])
                print("iter: %d" % n, "converge: %0.6f" % converge, "val-elbo: %0.5f" % ELBO_R[n])

if __name__ == '__main__':
    main()