# Import Orekit
import orekit
import os
try:
  __init_orekit
except:
  print("Importing JVM...")
  vm = orekit.initVM()
  print('Java version:', vm.java_version)
  print("Initializing Orekit...")
  from java.io import File
  from org.orekit.data import DataProvidersManager, ZipJarCrawler
  DM = DataProvidersManager.getInstance()
  datafile = File(os.path.join(os.path.dirname(__file__), '..', 'data', 'orekit-data.zip'))
  if not datafile.exists():
    raise 'File :' + datafile.absolutePath + ' not found'
  crawler = ZipJarCrawler(datafile)
  DM.clearProviders()
  DM.addProvider(crawler)
  __init_orekit = True
    
    
# Import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
  __init_matplotlib
except:
  plt.style.use('default')
  mpl.rcParams['axes.grid'] = True
  __init_matplotlib = True


# Impor Keras and Tensorflow
import keras
import tensorflow as tf
from keras import backend as K
from tensorflow.python.client import device_lib
try:
  __init_keras
except:
  #config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
  #sess = tf.compat.v1.Session(config=config) 
  sess = tf.compat.v1.Session()
  keras.backend.set_session(sess)
  print(device_lib.list_local_devices())
  print(K.tensorflow_backend._get_available_gpus())
  __init_keras = True
