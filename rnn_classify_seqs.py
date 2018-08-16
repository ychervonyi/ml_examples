import pandas as pd, numpy as np, tensorflow as tf
import blogs_data #available at https://github.com/spitis/blogs_data

df = blogs_data.loadBlogs().sample(frac=1).reset_index(drop=True)
vocab, reverse_vocab = blogs_data.loadVocab()
# train_len, test_len = np.floor(len(df)*0.8), np.floor(len(df)*0.2)
train_len, test_len = np.floor(len(df)*0.1), np.floor(len(df)*0.01)
train, test = df.ix[:train_len-1], df.ix[train_len:train_len + test_len]
print('Train length: ', train_len)
df = None

class SimpleDataIterator():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n
        return res['as_numbers'], res['gender']*3 + res['age_bracket'], res['length']

data = SimpleDataIterator(train)
d = data.next_batch(3)
print('Input sequences\n', d[0], end='\n\n')
print('Target values\n', d[1], end='\n\n')
print('Sequence lengths\n', d[2])

class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

        return x, res['gender']*3 + res['age_bracket'], res['length']

data = PaddedDataIterator(train)
d = data.next_batch(3)
print('Input sequences\n', d[0], end='\n\n')

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

vocab_size = len(vocab)
state_size = 64
batch_size = 256
num_classes = 6


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
# reset_graph()

# Placeholders
x = tf.placeholder(tf.int32, [batch_size, None]) # [batch_size, num_steps]
seqlen = tf.placeholder(tf.int32, [batch_size])
y = tf.placeholder(tf.int32, [batch_size])
keep_prob = tf.placeholder(tf.float32)

# Embedding layer
embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
tf.summary.histogram('embeddings', embeddings)
tf.summary.histogram('rnn_inputs', rnn_inputs)

# RNN
cell = tf.nn.rnn_cell.GRUCell(state_size)
init_state = tf.get_variable('init_state', [1, state_size],
                             initializer=tf.constant_initializer(0.0))
init_state = tf.tile(init_state, [batch_size, 1])
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen,
                                             initial_state=init_state)
tf.summary.histogram('rnn_outputs', rnn_outputs)
# Add dropout, as the model otherwise quickly overfits
rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
tf.summary.histogram('rnn_outputs_dropout', rnn_outputs)

"""
Obtain the last relevant output. The best approach in the future will be to use:

    last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), seqlen-1], axis=1))

which is the Tensorflow equivalent of numpy's rnn_outputs[range(30), seqlen-1, :], but the
gradient for this op has not been implemented as of this writing.

The below solution works, but throws a UserWarning re: the gradient.
"""
idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

tf.summary.histogram('rnn_outputs_last', last_rnn_output)

# Softmax layer
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    variable_summaries(b)
    variable_summaries(W)
logits = tf.matmul(last_rnn_output, W) + b
tf.summary.histogram('rnn_outputs_logits', logits)

preds = tf.nn.softmax(logits)
correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
tf.summary.scalar('accuracy', accuracy)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
tf.summary.scalar('cross_entropy', loss)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

batch_size, num_epochs, iterator = 256, 5, PaddedDataIterator

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('class_blog/train')
test_writer = tf.summary.FileWriter('class_blog/test')

init = tf.global_variables_initializer()
global_step = 0
with tf.Session() as sess:
    sess.run(init)
    tr = iterator(train)
    te = iterator(test)

    step, acc = 0, 0
    tr_losses, te_losses = [], []
    current_epoch = 0
    while current_epoch < num_epochs:
        step += 1
        global_step += 1
        batch = tr.next_batch(batch_size)
        feed = {x: batch[0], y: batch[1], seqlen: batch[2], keep_prob: 0.6}
        summary, accuracy_, _ = sess.run([merged, accuracy, train_step], feed_dict=feed)
        acc += accuracy_
        train_writer.add_summary(summary, global_step)

        if step % 100 == 0:
            print("Step", step)

        if tr.epochs > current_epoch:
            current_epoch += 1
            tr_losses.append(acc / step)
            step, acc = 0, 0

            #eval test set
            te_epoch = te.epochs
            while te.epochs == te_epoch:
                step += 1
                batch = te.next_batch(batch_size)
                feed = {x: batch[0], y: batch[1], seqlen: batch[2], keep_prob: 1.0}
                summary, accuracy_ = sess.run([merged, accuracy], feed_dict=feed)
                acc += accuracy_
                test_writer.add_summary(summary, global_step)

            te_losses.append(acc / step)
            step, acc = 0,0
            print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])
