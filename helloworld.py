from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
# Set parameters
learning_rate=0.01
training_iteration=30
batch_size=100
display_step=2

#input placeholders
x=tf.placeholder("float",shape=[None,784])
y=tf.placeholder("float",shape=[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

with tf.name_scope("WX_b") as scope:
    model=tf.nn.softmax(tf.matmul(x,W)+b)
w_h=tf.summary.histogram("weights",W)
b_h=tf.summary.histogram("biases",b)


with tf.name_scope("cost_function") as scope:
    cost_function=-tf.reduce_sum(y*tf.log(model))
    tf.summary.scalar("cost_function",cost_function)

with tf.name_scope("train") as scope:
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
init=tf.global_variables_initializer()
merged_summary_op=tf.summary.merge_all()

with tf.Session() as sess:
   sess.run(init)

   summary_writer=tf.summary.FileWriter('/Users/Priya',sess.graph)
   for iteration in range(training_iteration):
       avg_cost=0
       total_batch=int(mnist.train.num_examples/batch_size)
       for i in range(total_batch):
           batch_xs,batch_ys=mnist.train.next_batch(batch_size)
           sess.run(optimizer,feed_dict={x: batch_xs,y: batch_ys})
           avg_cost=sess.run(cost_function,feed_dict={x:batch_xs,y:batch_ys})/total_batch
           summary_str=sess.run(merged_summary_op,feed_dict={x:batch_xs,y:batch_ys})
           summary_writer.add_summary(summary_str,iteration*total_batch+i)
       if iteration % display_step == 0:
          print ("Iteration:",'%04d'%(iteration+1),"cost=","{:.9f}".format(avg_cost))
   print ("Tuning completed")
   predictions=tf.equal(tf.argmax(model,1),tf.argmax(y,1))
   accuracy=tf.reduce_mean(tf.cast(predictions,"float"))
   print(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
