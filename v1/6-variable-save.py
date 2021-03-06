import tensorflow as tf

weights = tf.Variable(tf.random_normal([784, 200], 
                                       stddev=0.35),
                      name="weights")

biases = tf.Variable(tf.zeros([200]), name="biases")

init_op = tf.initialize_all_variables()

# create the saver
saver = tf.train.Saver()

with tf.Session() as sess:
    # 1, run
    sess.run(init_op)
    
    print(sess.run(weights))
    print(sess.run(biases))
    
    # 2, save
    save_path = saver.save(sess, "./saver/model.ckpt")
    print(save_path)
    print("Model saved in: ", save_path)

