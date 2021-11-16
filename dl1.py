import tensorflow as tf
tf.compat.v1.disable_eager_execution()
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(5.0)
sess=tf.compat.v1.Session()

d=tf.multiply(node1,node2)
e=tf.add(d,node3)
print(sess.run(e))
sess.close()