import argparse
import numpy as np
import torch
import utils
import random
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from statistics import median

random.seed(94305)

def format_data_loader(data_loader):
    data_loader_format = []
    for image, label in data_loader:
        image = image.numpy()
        label = label.numpy()
        dig_list = []
        for d in range(10):
            digits = image[np.nonzero(label[0])]
            dig_list.append(digits)

        multi_mnist_sequences = []
        values = []
        low, high = 0, 10 ** l - 1
        for i in range(n):
            mnist_digits = []
            num = random.randint(low, high)
            values.append(num)

            for i in range(l):
                digit = num % 10
                num //= 10
                ref = dig_list[digit]
                print(ref.shape)
                mnist_digits.insert(0, ref[np.random.randint(0, ref.shape[0])])
            multi_mnist_sequence = np.concatenate(mnist_digits)
            multi_mnist_sequence = np.reshape(multi_mnist_sequence, (-1, 28))
            multi_mnist_sequences.append(multi_mnist_sequence)
        multi_mnist_batch = np.stack(multi_mnist_sequences)
        vals = np.array(values)
        med = int(median(values))
        arg_med = np.equal(vals, med).astype('float32')
        arg_med /= np.sum(arg_med)

        data_loader_format.append([multi_mnist_batch, med, arg_med, vals])
    
    return data_loader_format

parser = argparse.ArgumentParser()
parser.add_argument('--M', default=1, type=int, help='batch size')
parser.add_argument('--n', default=3, type=int, help='number of elements to compare at a time')
parser.add_argument('--l', default=4, type=int, help='number of digits')
parser.add_argument('--tau', default=5, type=int, help='temperature (dependent meaning)')
parser.add_argument('--method', default='deterministic_neuralsort', help='which method to use?')
parser.add_argument('--n_s', default=5, type=int, help='number of samples')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs to train')
parser.add_argument('--lr', default=1e-4, help='initial learning rate')

args = parser.parse_args()

n_s = args.n_s
NUM_EPOCHS = args.num_epochs
M = args.M
n = args.n
l = args.l
tau = args.tau
method = args.method
initial_rate = args.lr

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

n_samples = len(train_dataset) # n_samples 60000
train_size = int(len(train_dataset) * 0.8) # train_size 48000
val_size = n_samples - train_size # val_size 12000

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_data_loader = DataLoader(train_dataset, batch_size=M, shuffle=False)
val_data_loader = DataLoader(val_dataset, batch_size=M, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=M, shuffle=False)

format_train_data_loader = format_data_loader(train_data_loader)
format_val_data_loader = format_data_loader(val_data_loader)
format_test_data_loader = format_data_loader(test_data_loader)

#train_iterator, val_iterator, test_iterator = iter(train_data_loader), iter(val_data_loader), iter(test_data_loader)
#print(train_iterator.next())

# train_iterator ([100, 1, 28, 28], [100]) 

## tensorflow version
# train_iterator ([100, 3, 112, 28], [100,], [100,3], [100,3])

temperature = torch.tensor([tau], dtype=float)
experiment_id = 'sort-%s-M%d-n%d-l%d-t%d' % (method, M, n, l, tau * 10)
checkpoint_path = 'checkpoints/%s/' % experiment_id

# handle = tf.placeholder(tf.string, ())
# X_iterator = tf.data.Iterator.from_string_handle(
#     handle,
#     (tf.float32, tf.float32, tf.float32, tf.float32),
#     ((M, n, l * 28, 28), (M,), (M, n), (M, n))
# )

# X, y, median_scores, true_scores = X_iterator.get_next()
# true_scores = tf.expand_dims(true_scores, 2)
# P_true = util.neuralsort(true_scores, 1e-10)

# if method == 'vanilla':
#     representations = multi_mnist_cnn.deepnn(l, X, n)
#     concat_reps = tf.reshape(representations, [M, n * n])
#     fc1 = tf.layers.dense(concat_reps, n * n)
#     fc2 = tf.layers.dense(fc1, n * n)
#     P_hat_raw = tf.layers.dense(fc2, n * n)
#     P_hat_raw_square = tf.reshape(P_hat_raw, [M, n, n])

#     P_hat = tf.nn.softmax(P_hat_raw_square, dim=-1)  # row-stochastic!

#     losses = tf.nn.softmax_cross_entropy_with_logits_v2(
#         labels=P_true, logits=P_hat_raw_square, dim=2)
#     losses = tf.reduce_mean(losses, axis=-1)
#     loss = tf.reduce_mean(losses)

# if method == 'sinkhorn':
#     representations = multi_mnist_cnn.deepnn(l, X, n)
#     pre_sinkhorn = tf.reshape(representations, [M, n, n])
#     P_hat = sinkhorn_operator(pre_sinkhorn, temp=temperature)
#     P_hat_logit = tf.log(P_hat)

#     losses = tf.nn.softmax_cross_entropy_with_logits_v2(
#         labels=P_true, logits=P_hat_logit, dim=2)
#     losses = tf.reduce_mean(losses, axis=-1)
#     loss = tf.reduce_mean(losses)

# if method == 'gumbel_sinkhorn':
#     representations = multi_mnist_cnn.deepnn(l, X, n)
#     pre_sinkhorn = tf.reshape(representations, [M, n, n])
#     P_hat = sinkhorn_operator(pre_sinkhorn, temp=temperature)

#     P_hat_sample, _ = gumbel_sinkhorn(
#         pre_sinkhorn, temp=temperature, n_samples=n_s)
#     P_hat_sample_logit = tf.log(P_hat_sample)

#     P_true_sample = tf.expand_dims(P_true, 1)
#     P_true_sample = tf.tile(P_true_sample, [1, n_s, 1, 1])

#     losses = tf.nn.softmax_cross_entropy_with_logits_v2(
#         labels=P_true_sample, logits=P_hat_sample_logit, dim=3)
#     losses = tf.reduce_mean(losses, axis=-1)
#     losses = tf.reshape(losses, [-1])
#     loss = tf.reduce_mean(losses)

# if method == 'deterministic_neuralsort':
#     scores = multi_mnist_cnn.deepnn(l, X, 1)
#     scores = tf.reshape(scores, [M, n, 1])
#     P_hat = util.neuralsort(scores, temperature)

#     losses = tf.nn.softmax_cross_entropy_with_logits_v2(
#         labels=P_true, logits=tf.log(P_hat + 1e-20), dim=2)
#     losses = tf.reduce_mean(losses, axis=-1)
#     loss = tf.reduce_mean(losses)

# if method == 'stochastic_neuralsort':
#     scores = multi_mnist_cnn.deepnn(l, X, 1)
#     scores = tf.reshape(scores, [M, n, 1])
#     P_hat = util.neuralsort(scores, temperature)

#     scores_sample = tf.tile(scores, [n_s, 1, 1])
#     scores_sample += util.sample_gumbel([M * n_s, n, 1])
#     P_hat_sample = util.neuralsort(
#         scores_sample, temperature)

#     P_true_sample = tf.tile(P_true, [n_s, 1, 1])
#     losses = tf.nn.softmax_cross_entropy_with_logits_v2(
#         labels=P_true_sample, logits=tf.log(P_hat_sample + 1e-20), dim=2)
#     losses = tf.reduce_mean(losses, axis=-1)
#     loss = tf.reduce_mean(losses)
# else:
#     raise ValueError("No such method.")


# def vec_gradient(l):  # l is a scalar
#     gradient = tf.gradients(l, tf.trainable_variables())
#     vec_grads = [tf.reshape(grad, [-1]) for grad in gradient]  # flatten
#     z = tf.concat(vec_grads, 0)  # n_params
#     return z


# prop_correct = util.prop_correct(P_true, P_hat)
# prop_any_correct = util.prop_any_correct(P_true, P_hat)

# opt = tf.train.AdamOptimizer(initial_rate)
# train_step = opt.minimize(loss)
# saver = tf.train.Saver()

# # MAIN BEGINS

# sess = tf.Session()
# logfile = open('./logs/%s.log' % experiment_id, 'w')


# def prnt(*args):
#     print(*args)
#     print(*args, file=logfile)


# sess.run(tf.global_variables_initializer())
# train_sh, validate_sh, test_sh = sess.run([
#     train_iterator.string_handle(),
#     val_iterator.string_handle(),
#     test_iterator.string_handle()
# ])


# TRAIN_PER_EPOCH = mnist_input.TRAIN_SET_SIZE // (l * M)
# VAL_PER_EPOCH = mnist_input.VAL_SET_SIZE // (l * M)
# TEST_PER_EPOCH = mnist_input.TEST_SET_SIZE // (l * M)
# best_correct_val = 0


# def save_model(epoch):
#     saver.save(sess, checkpoint_path + 'checkpoint', global_step=epoch)


# def load_model():
#     filename = tf.train.latest_checkpoint(checkpoint_path)
#     if filename == None:
#         raise Exception("No model found.")
#     prnt("Loaded model %s." % filename)
#     saver.restore(sess, filename)


# def train(epoch):
#     loss_train = []
#     for _ in range(TRAIN_PER_EPOCH):
#         _, l = sess.run([train_step, loss],
#                         feed_dict={handle: train_sh})
#         loss_train.append(l)
#     prnt('Average loss:', sum(loss_train) / len(loss_train))


# def test(epoch, val=False):
#     global best_correct_val
#     p_cs = []
#     p_acs = []
#     for _ in range(VAL_PER_EPOCH if val else TEST_PER_EPOCH):
#         p_c, p_ac = sess.run([prop_correct, prop_any_correct], feed_dict={
#                              handle: validate_sh if val else test_sh,
#                              evaluation: True})
#         p_cs.append(p_c)
#         p_acs.append(p_ac)

#     p_c = sum(p_cs) / len(p_cs)
#     p_ac = sum(p_acs) / len(p_acs)

#     if val:
#         prnt("Validation set: prop. all correct %f, prop. any correct %f" %
#              (p_c, p_ac))
#         if p_c > best_correct_val:
#             best_correct_val = p_c
#             prnt('Saving...')
#             save_model(epoch)
#     else:
#         prnt("Test set: prop. all correct %f, prop. any correct %f" % (p_c, p_ac))


# for epoch in range(1, NUM_EPOCHS + 1):
#     prnt('Epoch', epoch, '(%s)' % experiment_id)
#     train(epoch)
#     test(epoch, val=True)
#     logfile.flush()
# load_model()
# test(epoch, val=False)

# sess.close()
# logfile.close()
