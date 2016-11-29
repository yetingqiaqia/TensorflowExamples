import math
import time

import tensorflow as tf
import input_data

tf.app.flags.DEFINE_string("data_dir", "MNIST_data",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string(
    "checkpoint_dir", "./checkpoint", "Checkpoint directory from training run")

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28

def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Variables of the hidden layer
  hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units], stddev=1.0 / IMAGE_PIXELS), name="hid_w")
  hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

  # Variables of the softmax layer
  sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10], stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="sm_w")
  sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

  x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
  y_ = tf.placeholder(tf.float32, [None, 10])

  hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
  hid = tf.nn.relu(hid_lin)

  y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
  loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

  global_step = tf.Variable(0)

  train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  summary_op = tf.merge_all_summaries()
  saver = tf.train.Saver()
  #Load model
  checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  print("checkpoint file: {}".format(checkpoint_file))

  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    time_begin = time.time()
    #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    #time_begin = time.time()
      
    # Loop until the supervisor shuts down or 3000 steps have completed.
    #local_step = 0
    #while True:
      #batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      #train_feed = {x: batch_xs, y_: batch_ys}

      #_, step = sess.run([train_op, global_step], feed_dict=train_feed)

      #if local_step % 100 == 0:
        # Validation feed
        #val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        #val_xent = sess.run(loss, feed_dict=val_feed)
        #print("%f: training step %d done (global step: %d) with loss %g" % (time.time(), local_step, step, val_xent))
       
      #local_step += 1

      #if val_xent < 500:
        #break
    print("Accuracy = %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    time_end = time.time()
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

    #print("Accuracy = %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
  tf.app.run()
