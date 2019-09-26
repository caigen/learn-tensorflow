"""
ref:
https://tensorflow.google.cn/tutorials/eager/automatic_differentiation
"""

import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

print("====Gradient Tapes====")
print("non-persistent tapes")
x = tf.ones((2, 2))
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        print(dz_dx[i][j].numpy())

print("non-persistent tapes")
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
dz_dy = t.gradient(z, y)
print(dz_dy.numpy())

print("persistent tapes")
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x * x
    z = y * y

dy_dx = t.gradient(y, x)    # x ^ 2  => 2 * x
dz_dx = t.gradient(z, x)    # x ^ 4  => 4 * x ^ 3
print(dy_dx.numpy(), dz_dx.numpy())
del t

print("====recording control flow===")
def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)
print(grad(x, 6).numpy())   # x ^ 3 => 6 * x
print(grad(x, 5).numpy())   # x ^ 3 => 6 * x
print(grad(x, 4).numpy())   # x * x => 2 * x

print("higher-order gradients")
x = tf.Variable(1.0)

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    dy_dx = t2.gradient(y, x) # x ^ 3 => 3 * x ^ 2
    d2y_dx2 = t.gradient(dy_dx, x) # => 6 * x

print(dy_dx.numpy(), d2y_dx2.numpy())