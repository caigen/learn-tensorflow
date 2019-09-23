import tensorflow as tf

# Variable for state
state = tf.Variable(0, name="counter")

# add op
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# init op
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)

    print(sess.run(state))

    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

