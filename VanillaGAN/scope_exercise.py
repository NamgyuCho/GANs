# see https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow

import tensorflow as tf

"""
with tf.name_scope('my_scope'):
    v1 = tf.get_variable('var1', [1], dtype=tf.float32)
    v2 = tf.Variable(1, name='var2', dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)
print(v2.name)
print(a.name)

with tf.variable_scope('my_scope'):
    v1 = tf.get_variable('var1', [1], dtype=tf.float32)
    v2 = tf.Variable(1, name='var2', dtype=tf.float32)
    a = tf.add(v1, v2)

print('\n\n')
print(v1.name)
print(v2.name)
print(a.name)
"""

with tf.name_scope('foo'):
    with tf.variable_scope('var_scope'):
        v = tf.get_variable('var', [1])

with tf.name_scope('bar'):
    with tf.variable_scope('var_scope', reuse=True):
        v1 = tf.get_variable('var', [1])
assert v == v1

print(v.name)
print(v1.name)
