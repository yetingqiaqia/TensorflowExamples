import math
import time

import tensorflow as tf
import input_data

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

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
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  print("ps_hosts: "+FLAGS.ps_hosts)
  print("worker_hosts: "+FLAGS.worker_hosts)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    is_chief = (FLAGS.task_index == 0)
    workers_number = len(FLAGS.worker_hosts)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
      # Variables of the hidden layer
      #tf.truncated_normal: Outputs random values from a truncated normal distribution.
      hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units], stddev=1.0 / IMAGE_PIXELS), name="hid_w")
      hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

      # Variables of the softmax layer
      sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10], stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name="sm_w")
      sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
      y_ = tf.placeholder(tf.float32, [None, 10])

      #tf.nn.xw_plus_b(x, w, b) equals to tf.matmul(x, w)+b
      #hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      hid_lin = tf.matmul(x, hid_w) + hid_b
      #tf.nn.relu(features): max(features, 0), https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
      hid = tf.nn.relu(hid_lin)

      y = tf.nn.softmax(tf.matmul(hid, sm_w)+sm_b)
      #y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
      #tf.clip_by_value(t, clip_value_min, clip_value_max): Clips tensor values to a specified min and max.
      loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

      saver = tf.train.Saver()
      if is_chief:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=FLAGS.checkpoint_dir,
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=60)

    # The supervisor takes care of session initialization, restoring from a checkpoint, and closing when done or an error occurs.
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = sv.prepare_or_wait_for_session(master=server.target, config=sess_config)
    time_begin = time.time()
    # Loop until the supervisor shuts down or 3000 steps have completed.
    #saver = tf.train.Saver()
    local_step = 0
    while True:
      # Run a training step asynchronously.
      # See `tf.train.SyncReplicasOptimizer` for additional details on how to
      # perform *synchronous* training.

      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      train_feed = {x: batch_xs, y_: batch_ys}

      _, step = sess.run([train_op, global_step], feed_dict=train_feed)

      if local_step % 100 == 0:
        # Validation feed
        val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        val_xent = sess.run(loss, feed_dict=val_feed)
        print("%f: Worker %d: training step %d done (global step: %d) with loss %g" % (time.time(), FLAGS.task_index, local_step, step, val_xent))
        #if is_chief:
            #save_path = saver.save(sess, "./output/model.ckpt-mnist-"+str(local_step/100))
            #print("Model saved in file: %s" % save_path)

      local_step+=1

      if val_xent < 340 or local_step > 200000/workers_number:
        break

    time_end = time.time()
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

    if is_chief:
      print("Accuracy = %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
  tf.app.run()
