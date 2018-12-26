import tensorflow as tf
from model_aon import inference, get_train_op, get_init_op
from input_data import get_batch_data
from input_data import get_inputdata
import os
import numpy as np
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"

flags = tf.app.flags
flags.DEFINE_string('exp_dir', '/search/odin/xjz/OCR_Recognition/AON-testGPU/AON-master/exp_log_FG_twoGPU',
                    'experiment model save directory')
flags.DEFINE_integer('batch_size_per_gpu', 128, 'define train batch size')
flags.DEFINE_integer('max_steps', 300000, 'step nums for training')
flags.DEFINE_integer('num_epochs', 30, 'step nums for training')
flags.DEFINE_boolean('restore', True, 'restore model parameter from checkpoint file')
flags.DEFINE_string('tfrecord_file_path', '/search/odin/xjz/Data/huawei_test.tfrecord', 'tfrecord file path')
flags.DEFINE_boolean('single_seq', True, 'Use FG or not')
tf.app.flags.DEFINE_string('gpu_list', '5,7', '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
FLAGS = flags.FLAGS

gpulist = list(range(len(FLAGS.gpu_list.split(','))))


def average_gradients(tower_grads):
    average_grads = []
    counts = 0

    for grad_and_vars in zip(*tower_grads):
        grads = []

        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_loss(images, groundtruth, single_seq=True, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        output_tensor_dict, _ = inference(images, groundtruth, single_seq)
    loss = output_tensor_dict['loss']
    return loss


def main(unused_argv):
    if FLAGS.exp_dir:
        checkpoint_dir = os.path.join(FLAGS.exp_dir, 'model.ckpt')
        train_log_write_dir = os.path.join(FLAGS.exp_dir, 'log/train')

    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.name_scope('input'):
        image_placeholder = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32)
        groundtruth_text_placeholder = tf.placeholder(shape=[None, ], dtype=tf.string)
        tf.summary.image('input_image', image_placeholder, FLAGS.batch_size_per_gpu)
    print('image_placeholder', image_placeholder)
    print('groundtruth_placeholder', groundtruth_text_placeholder)
    
    output_tensor_dict, _ = inference(image_placeholder, groundtruth_text_placeholder, FLAGS.single_seq)
    output_predict_text_tensor = output_tensor_dict['predict_text']


    opt = tf.train.AdadeltaOptimizer(learning_rate=1.0)

    tower_grads = []
    reuse_variables = True
    images, groundtruth = get_inputdata(FLAGS.tfrecord_file_path, FLAGS.num_epochs, batch_size=FLAGS.batch_size_per_gpu)

    for i, gpu_id in enumerate(gpulist):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('GPU_%d' % gpu_id)as scope:
                print ("gpu_id= ", gpu_id)
                loss = get_loss(images, groundtruth, reuse_variables=reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True
                grads = opt.compute_gradients(loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True


    batch_tensor_dict = get_batch_data(FLAGS.tfrecord_file_path, mode='train',batch_size=FLAGS.batch_size_per_gpu * len(gpulist))


    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(train_log_write_dir, sess.graph)
        summary_merge_tensor = tf.summary.merge_all()
        sess.run(get_init_op())
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        begin_step = 0
        if os.path.exists(os.path.join(FLAGS.exp_dir, 'checkpoint')) and FLAGS.restore:
            save_path = tf.train.latest_checkpoint(FLAGS.exp_dir)
            saver.restore(sess, save_path=save_path)
            begin_step = sess.run(global_step)
            print('Restore model from {} successful, continue training from step {}'.format(save_path, begin_step))
        total_loss = 0
        for step in range(begin_step, FLAGS.max_steps):
            start_time = time.time()
            _, loss_ = sess.run([train_op, loss])
            duration = time.time() - start_time
            total_loss += loss_
            # print('Step {}, loss {}'.format(step, loss_))
            if step % 10 == 0:
                num_example_per_step = FLAGS.batch_size_per_gpu * len(gpulist)
                examples_per_sec = num_example_per_step / duration
                sec_per_batch = duration
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_, examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                batch_dict = sess.run(batch_tensor_dict)
                images_test = batch_dict['images']
                groundtruth_text_test = np.char.lower(batch_dict['groundtruth_text'].astype('str'))
                feed_dict = {image_placeholder: images_test, groundtruth_text_placeholder: groundtruth_text_test}
                summary, output_predict_text, = sess.run([summary_merge_tensor, output_predict_text_tensor], feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                acc_num = 0
                for i in range(FLAGS.batch_size_per_gpu * len(gpulist)):
                    if output_predict_text[i] == groundtruth_text_test[i]:
                        acc_num += 1
                accrruracy = float(acc_num) / (FLAGS.batch_size_per_gpu * len(gpulist))
                
                num_example_per_step = FLAGS.batch_size_per_gpu * len(gpulist)
                examples_per_sec = num_example_per_step / duration
                sec_per_batch = duration
                format_str = ('%s: step %d, accrruracy = %.2f')
                print(format_str % (datetime.now(), step, accrruracy))
            if step % 10000 == 0:
                saver.save(sess, save_path=checkpoint_dir, global_step=global_step)
                print('Write checkpoint {}'.format(sess.run(global_step)))

        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    tf.app.run()

