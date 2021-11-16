import tensorflow as tf
tf.compat.v1.disable_eager_execution()

a=tf.compat.v1.placeholder(tf.float32)
b=tf.compat.v1.placeholder(tf.float32)
c=a+b
s=tf.compat.v1.Session()
print(s.run(c,{a:[1,3],b:[2,4]}))