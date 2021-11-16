import tensorflow as tf
tf.compat.v1.disable_eager_execution()
w=tf.Variable([.3],tf.float32)
b=tf.Variable([-.3],tf.float32)
x=tf.compat.v1.placeholder(tf.float32)
lm=w*x+b
init=tf.compat.v1.global_variables_initializer()
sess=tf.compat.v1.Session()
sess.run(init)
#print(sess.run(lm,{x:[1,2,3,4]}))

y=tf.compat.v1.placeholder(tf.float32)
sq_dt=tf.square(lm-y)
loss=tf.reduce_sum(sq_dt)
#print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
op=tf.compat.v1.train.GradientDescentOptimizer(0.01)
train=op.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
print(sess.run([w,b]))