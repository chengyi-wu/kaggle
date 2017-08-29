import tensorflow as tf
import logging
import numpy as np
from utils import Progbar
from model import Model
from data_utils import load_data

class CATSModel(Model):
    def __init__(self):
        self.input_placeholder = None
        self.labels_placeholder = None
        self.build()

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='X')
        self.labels_placeholder = tf.placeholder(dtype=tf.int64, shape=(None), name='y')

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {
            self.input_placeholder : inputs_batch,
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        '''
        Setup the VGG model here
        ''' 
        a = tf.contrib.layers.conv2d(
            inputs=self.input_placeholder,
            num_outputs=64,
            kernel_size=3,
            stride=1,
            activation_fn=None
        )
        h = tf.nn.relu(a)
        h = tf.contrib.layers.max_pool2d(
            inputs=h,
            kernel_size=2,
            stride=2,
        )

        a = tf.contrib.layers.conv2d(
            inputs=h,
            num_outputs=64,
            kernel_size=3,
            stride=1,
            activation_fn=None
        )
        h = tf.nn.relu(a)
        h = tf.contrib.layers.max_pool2d(
            inputs=h,
            kernel_size=2,
            stride=2,
        )
        
        h_flat = tf.reshape(h, shape=(-1, 8 * 8 * 64))

        y_out = tf.contrib.layers.fully_connected(
            inputs=h_flat,
            num_outputs=2,
            activation_fn=None
        )
        
        return y_out

    def add_loss_op(self, y_out):
        y = self.labels_placeholder

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(y, 2),
            logits=y_out
        )
        loss = tf.reduce_mean(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        return loss, correct_prediction, accuracy
    
    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, is_training=True):
        feed = self.create_feed_dict(inputs_batch, labels_batch)
        variables = [self.loss, self.correct_prediction, self.train_op]
        if not is_training:
            variables[-1] = self.accuracy
        loss, corr, _ = sess.run(
            variables, feed_dict=feed
        )
        return loss, corr

    def run_epoch(self, sess, batch_size, training_set, is_training=True):
        X_tr, Y_tr = training_set

        prog = Progbar(target=1 + X_tr.shape[0] // batch_size)
        
        total_correct, total_items = 0, 0

        for i, (train_x, train_y) in enumerate(get_minibatches(X_tr, Y_tr, batch_size)):
            
            loss, corr = self.train_on_batch(sess, train_x, train_y, is_training)
            total_correct += np.sum(corr)
            total_items += train_y.shape[0]
            if is_training:
                prog.update(i + 1, [('train loss', loss)], [('train accuracy', total_correct / total_items)])
            else:
                prog.update(i + 1, None, [('validation accuracy', total_correct / total_items)])

        
    def fit(self, sess, epoches, batch_size, training_set, validation_set):
        for epoch in range(epoches):
            print("\nEpch {:} out of {:}".format(epoch + 1, epoches))
            self.run_epoch(sess, batch_size, training_set)
            self.run_epoch(sess, batch_size, validation_set, is_training=False)

def get_minibatches(data, labels, minibatch_size, shuffle=True):
    data_size = data.shape[0]
    indicies = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indicies)
    for start_idx in range(0, data_size, minibatch_size):
        idx = indicies[start_idx : start_idx + minibatch_size]
        yield data[idx], labels[idx]

def main(debug=True):
    X_tr, Y_tr, X_val, Y_val, X_te = load_data("datasets")

    tf.reset_default_graph()

    model = CATSModel()

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            sess.run(tf.global_variables_initializer())
            model.fit(
                sess,
                1,
                32,
                (X_tr, Y_tr),
                (X_val, Y_val)
            )
    print("")

if __name__=='__main__':
    main()