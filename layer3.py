import argparse
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
tfSession = tf.InteractiveSession()

# Parameters
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

argsParser = argparse.ArgumentParser()
argsParser.add_argument('-b', default=100, type=int, dest='batch_size')
argsParser.add_argument('-e', default=10000, type=int, dest='epochs')
argsParser.add_argument('-l', default=.2, type=float, dest='learning_rate')
argsParser.add_argument('-o', default=1, type=int, choices=range(1, len(optimizers)+1), dest='optimizer')
argsParser.add_argument('-p', default=32, type=int, choices=(32, 64), dest='precision')
argsParser.add_argument('-s', default=0, type=int, dest='random_seed')
args = argsParser.parse_args()

dtype = np.float32 if args.precision==32 else np.float64
np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)

# Random number generator that returns the correct type
# and returns the same sequence regardless of type
def rndTypeIndependent(shape=(), **kwargs):
  """shape: the shape of the random tensor"""
  if type(shape)==int or type(shape)==float:
    shape = shape,
  if 'stddev' not in kwargs and type(shape) in (list, tuple) and shape:
    kwargs['stddev'] = math.sqrt(shape[0])
  x = tf.random_normal(shape, **kwargs, dtype=np.float64)
  if dtype==np.float32:
    x = tf.to_float(x)
  return x.eval()

# Output to screen and log file
logFile = open(os.path.splitext(os.path.basename(__file__))[0]+'.log', 'a')

def printToLog(x='', *args):
  if args:
    x = x.format(*args)
  else:
    x = str(x)
  logFile.write(x)
  logFile.write('\n')
  print(x)

printToLog()
printToLog('TensorFlow {}, Python {}, {}',
  tf.__version__,
  sys.version,
  platform.platform())
printToLog('{:14} {:10} bytes, {}, {} precision, batch size {}, learning rate {}, random seed {}',
  os.path.basename(__file__),
  os.path.getsize(__file__),
  time.strftime('%b %d %Y, %H:%M:%S', time.localtime(os.path.getmtime(__file__))),
  'single' if dtype==np.float32 else 'double',
  args.batch_size,
  args.learning_rate,
  args.random_seed
  )

# Data
def loadCsvFile(filename):
  data = np.loadtxt(filename, dtype=dtype)
  printToLog('{:14} {:10} bytes, {}, {} rows, {} columns',
    filename,
    os.path.getsize(filename),
    time.strftime('%b %d %Y, %H:%M:%S', time.localtime(os.path.getmtime(filename))),
    np.shape(data)[0],
    np.shape(data)[1]
    )
  return data

training_data = loadCsvFile('training.dat')

validation_data = loadCsvFile('validation.dat')
validation_Y = validation_data[:, 0]
validation_X = validation_data[:, 1:]

columnCount = np.shape(validation_X)[1]

# Inputs and outputs
X = tf.placeholder(dtype, shape=(None, columnCount))
Y = tf.placeholder(dtype, shape=None,)

# Hidden layers
n1 = 4
w1 = tf.Variable(rndTypeIndependent((columnCount, n1)), dtype=dtype)
b1 = tf.Variable(rndTypeIndependent(n1), dtype=dtype)
a1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1)
printToLog('layer 1: {}', n1)

n2 = 3
w2 = tf.Variable(rndTypeIndependent((n1, n2)), dtype=dtype)
b2 = tf.Variable(rndTypeIndependent(n2), dtype=dtype)
a2 = tf.nn.sigmoid(tf.matmul(a1, w2) + b2)
printToLog('layer 2: {}', n2)

# Output layer
no = 1
wo = tf.Variable(rndTypeIndependent((n2, no)), dtype=dtype)
bo = tf.Variable(rndTypeIndependent(no), dtype=dtype)
p = tf.nn.sigmoid(tf.squeeze(tf.matmul(a2, wo)) + bo)
tf.global_variables_initializer().run()

# Model
cost = tf.reduce_sum((p-Y)**2/np.shape(training_data)[0])

# Optimizer
def getTFOptimizer(i):
  """i: the optimiser number"""
  if i==1:
    return tf.train.AdadeltaOptimizer(args.learning_rate)
  if i==2:
    return tf.train.AdagradOptimizer(args.learning_rate)
  if i==3:
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
    return tf.train.AdagradDAOptimizer(args.learning_rate, global_step=global_step)
  if i==4:
    return tf.train.AdamOptimizer(args.learning_rate)
  if i==5:
    return tf.train.FtrlOptimizer(args.learning_rate)
  if i==6:
    return tf.train.GradientDescentOptimizer(args.learning_rate)
  if i==7:
    momentum = 1.0
    return tf.train.MomentumOptimizer(args.learning_rate, momentum)
  if i==8:
    return tf.train.ProximalAdagradOptimizer(args.learning_rate)
  if i==9:
    return tf.train.ProximalGradientDescentOptimizer(args.learning_rate)
  if i==10:
    return tf.train.RMSPropOptimizer(args.learning_rate)

optimizer = getTFOptimizer(args.optimizer).minimize(cost)
tf.global_variables_initializer().run()

# Classifier
classify = tf.greater(p, 0.5)
correct = tf.reduce_sum(tf.cast(tf.equal(tf.cast(classify, dtype), Y), tf.int32))

# Train
printToLog()
printToLog(optimizers[args.optimizer-1])
printToLog(datetime.datetime.now().strftime('%a, %b %d, %Y, %H:%M:%S'))
printToLog('{:>6} {:>18} {:>18} {:>18}',
  'epoch',
  'training_cost',
  'training_acc',
  'validation_acc'
  )
start_time = time.time()
for epoch in range(args.epochs):
  np.random.shuffle(training_data)
  training_Y = training_data[:, 0]
  training_X = training_data[:, 1:]

  # Print update at successive doublings of time
  if epoch&(epoch-1)==0 or epoch==args.epochs-1:
    printToLog('{:6} {:18.16f} {:18.16f} {:18.16f}',
      epoch,
      cost.eval({X: training_X, Y: training_Y}),
      float(correct.eval({X: training_X, Y: training_Y}))/np.shape(training_data)[0],
      float(correct.eval({X: validation_X, Y: validation_Y}))/np.shape(validation_data)[0]
      )

  for i in range(0, np.shape(training_data)[0], args.batch_size):
    batch_X = training_X[i: i+args.batch_size]
    batch_Y = training_Y[i: i+args.batch_size]
    optimizer.run({X: batch_X, Y: batch_Y})

# Test
filename = 'test.dat'
printToLog()
printToLog('{:14} {:10} bytes, {}',
  filename,
  os.path.getsize(filename),
  time.strftime('%b %d %Y, %H:%M:%S', time.localtime(os.path.getmtime(filename)))
  )
confusion_matrix = [[0, 0], [0, 0]]
reader = csv.reader(open(filename), delimiter=' ')
for r in reader:
  test_Y = [r[0]]
  test_X = [r[1:]]
  predicted = int(classify.eval({X: test_X, Y: test_Y}))
  actual = int(r[0])
  confusion_matrix[predicted][actual] += 1
printToLog('{:9} {:6} {}',
  '',
  '',
  'Actual'
  )
printToLog('{:9} {:6} {:>6} {:>6}',
  '',
  '',
  'True',
  'False'
  )
printToLog('{:9} {:>6} {:6} {:6}',
  'Predicted',
  'True',
  confusion_matrix[1][1],
  confusion_matrix[1][0]
  )
printToLog('{:9} {:>6} {:6} {:6}',
  '',
  'False',
  confusion_matrix[0][1],
  confusion_matrix[0][0]
  )

# Resource summary
printToLog()
printToLog('{} bytes', process.memory_info().peak_wset)
printToLog('{} seconds', time.time()-start_time)
