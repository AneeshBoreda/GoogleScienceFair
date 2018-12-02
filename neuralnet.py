

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm
import random
class Model:
    def __init__(self, data_x, data_y):
        self.n_class = 7
        self._create_architecture(data_x, data_y)

    def _create_architecture(self, data_x, data_y):
        y_hot = tf.one_hot(data_y, depth = self.n_class)
        logits = self._create_model(data_x)
        predictions = tf.argmax(logits, 1, output_type = tf.int32)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_hot,
                                                                              logits = logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, data_y), tf.float32))

    def _create_model(self, X):
        X1 = X - 0.5
        #X1 = tf.pad(X1, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer = tf.truncated_normal_initializer(0.0, 0.1)):
            net = slim.conv2d(X1, 6, [5, 5], padding = 'VALID')
            net = slim.max_pool2d(net, [2, 2])
            net = slim.conv2d(net, 16, [5, 5], padding = 'VALID')
            net = slim.max_pool2d(net, [2, 2])

            net = tf.reshape(net, [-1, 256])
            net = slim.fully_connected(net, 120)
            net = slim.fully_connected(net, 84)
            net = slim.fully_connected(net, self.n_class, activation_fn = None)
        return net




def getLabels(x,image_to_label,labeldict):

    image_to_label['./Images/'+x[0].strip()+'.jpg']=int(labeldict[x[1].strip()])
    return x
def main(unused_argv):
   labeldict={}
   random.seed(9876)
   tf.reset_default_graph()
   X=tf.placeholder(tf.string,[None])
   Y=tf.placeholder(tf.int32,[None])
   num_epochs=1
   size=10015
   train=0.9
   test=0.1
   val=0.02
   batch=10
   train_size=int(train*size)
   test_size=int(test*size)
   val_size=int(val*size)
   labeldict['nv']=0
   labeldict['mel']=1
   labeldict['bkl']=2
   labeldict['bcc']=3
   labeldict['akiec']=4
   labeldict['vasc']=5
   labeldict['df']=6
   image_to_label={}
   #tf.enable_eager_execution()
   # Load training and eval data
   # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
   # A vector of filenames.
   dir_path = "./Images/"
   pictures = tf.gfile.ListDirectory(dir_path)
   #print(pictures)
   #exit(0)
   pictures = [dir_path+x.strip() for x in pictures]
   random.shuffle(pictures)
   filenames = tf.constant(pictures)
   print(filenames.shape)
   meta_data=np.genfromtxt('HAM10000_metadata.csv',delimiter=',',usecols=(1,2),dtype=np.unicode_,skip_header=1)
   np.apply_along_axis(func1d=getLabels,axis=1,arr=meta_data,image_to_label=image_to_label,labeldict=labeldict)
   # `labels[i]` is the label for the image in `filenames[i].
   #labels = tf.constant([image_to_label[i] for i in pictures])
   labels=[image_to_label[i] for i in pictures]
   train_x=pictures[0:train_size]
   train_y=labels[0:train_size]
   val_x=pictures[train_size:train_size+val_size]
   val_y=labels[train_size:train_size+val_size]
   test_x=pictures[train_size:]
   test_y=labels[train_size:]


   final_data=tf.data.Dataset.from_tensor_slices((X,Y))
   final_data=final_data.map(_parse_function)
   #dataset.batch(batch_size = 10)
   final_data=final_data.batch(batch)
   iter = final_data.make_initializable_iterator()

#   record_defaults = [tf.float32] * 2
   #meta_data = tf.contrib.data.CsvDataset(["HAM10000_metadata.csv"], record_defaults, select_cols = [2, 3], header = True)



  # for x,y in meta_data:
    #   print(x,y)
#   print(meta_data[1])
   #iter2 = meta_data.make_one_shot_iterator()
#   for x,y in iter2:
#      print(x,y)
   data_X,data_Y = iter.get_next()
   #data_Y=tf.cast(data_Y,tf.int32)
   print(data_X.shape)
   model= Model(data_X,data_Y)
   with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(num_epochs):
          train_loss, train_accuracy = 0, 0
          val_loss, val_accuracy = 0, 0
          sess.run(iter.initializer,feed_dict={X:train_x,Y:train_y})
          try:
               with tqdm(total = len(train_x)) as pbar:
                   while True:
                       #c+=1
                       _, loss, acc = sess.run([model.optimizer, model.loss, model.accuracy])
                       train_loss += loss
                       train_accuracy += acc
                       pbar.update(batch)
                   #if(c%1000==0):
                   #   print('Current Time:',time.time()-start)
                   #  print('Iteration:',c)
                       #print('Values:',data_Y)
          except tf.errors.OutOfRangeError:
              print('Training Loss:',loss)
              print('Training accuracy:',acc)
              print('Done')
          sess.run(iter.initializer,feed_dict={X:val_x,Y:val_y})
          try:
              while True:
                  loss, acc = sess.run([model.loss, model.accuracy])
                  val_loss += loss
                  val_accuracy += acc
          except tf.errors.OutOfRangeError:
              print('Validation Loss:',loss)
              print('Validation accuracy:',acc)
              print('Done')
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_X(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    image_resized = tf.image.rgb_to_grayscale(image_resized)
    return image_resized
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    image_resized = tf.image.rgb_to_grayscale(image_resized)

    return image_resized, label

"""train_data = mnist.train.images  # Returns np.array
   train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
   eval_data = mnist.test.images  # Returns np.array
   eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
   mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=10,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)"""



if __name__ == "__main__":
    tf.app.run()
