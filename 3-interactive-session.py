# for ipython
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# =====op: run()====
# initializer op: run() method
x.initializer.run()

# subtract op
sub = tf.subtract(x, a)

# =====tensor: eval()====
# eval() method
print(sub.eval())
