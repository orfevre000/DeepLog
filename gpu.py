import tensorflow as tf
from tensorflow.python.client import device_lib

if tf.test.is_gpu_available():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    print("GPUの読み込みが完了しました")

else:
    print("GPUが存在していません")
    device_lib.list_local_devices()
