from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from tensorflow.contrib import rnn
import csv
import datetime
import math
import numpy as np
import os
import platform
import psutil
import sys
import tensorflow as tf
import time
process = psutil.Process(os.getpid())

batch_size=100
epochs=1000
start_time = time.time()

optimizers = [
  'AdadeltaOptimizer',
  'AdagradOptimizer',
  'AdagradDAOptimizer',
  'AdamOptimizer',
  'FtrlOptimizer',
  'GradientDescentOptimizer',
  'MomentumOptimizer',
  'ProximalAdagradOptimizer',
  'ProximalGradientDescentOptimizer',
  'RMSPropOptimizer']

# Optimizer
def getTFOptimizer(i,learning_rate):
  if i=='AdadeltaOptimizer':
    return tf.train.AdadeltaOptimizer(learning_rate)
  if i=='AdagradOptimizer':
    return tf.train.AdagradOptimizer(learning_rate)
  if i=='AdagradDAOptimizer':
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
    return tf.train.AdagradDAOptimizer(learning_rate, global_step=global_step)
  if i=='AdamOptimizer':
    return tf.train.AdamOptimizer(learning_rate)
  if i=='FtrlOptimizer':
    return tf.train.FtrlOptimizer(learning_rate)
  if i=='GradientDescentOptimizer':
    return tf.train.GradientDescentOptimizer(learning_rate)
  if i=='MomentumOptimizer':
    momentum = 1.0
    return tf.train.MomentumOptimizer(learning_rate, momentum)
  if i=='ProximalAdagradOptimizer':
    return tf.train.ProximalAdagradOptimizer(learning_rate)
  if i=='ProximalGradientDescentOptimizer':
    return tf.train.ProximalGradientDescentOptimizer(learning_rate)
  if i=='RMSPropOptimizer':
    return tf.train.RMSPropOptimizer(learning_rate)

# Output to screen and log file
now = datetime.datetime.now()
logFile = open(now.strftime('%Y-%m-%dT%H_%M_%S')+'.log', 'a')

def printToLog(x='', *args):
  if args:
    x = x.format(*args)
  else:
    x = str(x)
  logFile.write(x)
  logFile.write('\n')
  print(x)

printToLog('TensorFlow {}, Python {}, {}',
  tf.__version__,
  sys.version,
  platform.platform())
printToLog('{:14} {:10} bytes, {}, batch size {}',
  os.path.basename(__file__),
  os.path.getsize(__file__),
  time.strftime('%b %d %Y, %H:%M:%S', time.localtime(os.path.getmtime(__file__))),
  batch_size,
  )

# Data
def loadCsvFile(filename):
  data = np.loadtxt(filename, dtype=np.float32)
  if len(np.shape(data))==2:
    printToLog('{:14} {:10} bytes, {}, {} rows, {} columns',
      filename,
      os.path.getsize(filename),
      time.strftime('%b %d %Y, %H:%M:%S', time.localtime(os.path.getmtime(filename))),
      np.shape(data)[0],
      np.shape(data)[1]
      )
  else:
    printToLog('{:14} {:10} bytes, {}, {} columns',
      filename,
      os.path.getsize(filename),
      time.strftime('%b %d %Y, %H:%M:%S', time.localtime(os.path.getmtime(filename))),
      np.shape(data)[0]
      )
  return data

shape_data = loadCsvFile('shape.dat')
length = int(shape_data[0])
channels = int(shape_data[1])
printToLog('length:          {}', length)
printToLog('channels:        {}', channels)

training_data = loadCsvFile('training.dat')

validation_data = loadCsvFile('validation.dat')
validation_Y = validation_data[:, 0]
validation_X = validation_data[:, 1:]

columnCount = np.shape(validation_X)[1]

space  = [
  Real(10**-3, 10**0, "log-uniform", name='learning_rate'),
  Integer(1,100, name='num_units'),
  Categorical(optimizers, name='optim'),
]

def fitness(gp_args):
  learning_rate,num_units,optim=gp_args
  learning_rate=float(learning_rate)
  printToLog()
  printToLog(datetime.datetime.now().strftime('%a, %b %d, %Y, %H:%M:%S'))
  printToLog('gp_args:         {}', gp_args)
  printToLog('learning_rate:   {}', learning_rate)
  printToLog('num_units:       {}', num_units)
  printToLog(optim)
  tf.reset_default_graph()
  sess = tf.InteractiveSession()

  # Inputs and outputs
  X = tf.placeholder(tf.float32, shape=(None, columnCount))
  Y = tf.placeholder(tf.float32, shape=None,)

  # Layers
  input_layer = tf.reshape(X, [-1, length, channels])

  #define constants
  time_steps=length
  n_input=channels
  n_classes=1

  #weights and biases of appropriate shape to accomplish above task
  out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
  out_bias=tf.Variable(tf.random_normal([n_classes]))

  #processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
  input=tf.unstack(input_layer ,time_steps,1)

  #defining the network
  lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
  outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

  #converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
  prediction=tf.squeeze(tf.matmul(outputs[-1],out_weights)+out_bias)

  #loss_function
  cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=Y))

  #model evaluation
  #correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
  #accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

  optimizer = getTFOptimizer(optim,learning_rate).minimize(cost)
  tf.global_variables_initializer().run()

  # Classifier
  classify = tf.greater(prediction, 0.5)
  correct = tf.reduce_sum(tf.cast(tf.equal(tf.cast(classify, tf.float32), Y), tf.int32))

  # Train
  printToLog('{:>6} {:>18} {:>18} {:>18}',
    'epoch',
    'training_cost',
    'training_acc',
    'validation_acc'
    )
  for epoch in range(epochs):
    np.random.shuffle(training_data)
    training_Y = training_data[:, 0]
    training_X = training_data[:, 1:]

    # Print update at successive doublings of time
    if epoch&(epoch-1)==0 or epoch==epochs-1:
      printToLog('{:6} {:18.16f} {:18.16f} {:18.16f}',
        epoch,
        cost.eval({X: training_X, Y: training_Y}),
        float(correct.eval({X: training_X, Y: training_Y}))/np.shape(training_data)[0],
        float(correct.eval({X: validation_X, Y: validation_Y}))/np.shape(validation_data)[0],
        )

    for i in range(0, np.shape(training_data)[0], batch_size):
      batch_X = training_X[i: i+batch_size]
      batch_Y = training_Y[i: i+batch_size]
      optimizer.run({X: batch_X, Y: batch_Y})

  printToLog('{} bytes', process.memory_info().peak_wset)
  return -correct.eval({X: validation_X, Y: validation_Y})

res=gp_minimize(fitness, space)
printToLog(res)
