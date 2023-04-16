import pyaudio
import librosa
import numpy as np
import tensorflow as tf

hop_length = 1024
n_fft = 1024
n_mels = 48
n_mfcc = 13
sr = 16000
num_frames = 128 * 1024

audio = pyaudio.PyAudio()

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]["quantization"]
input_shape = input_details[0]['shape']
output_scale, output_zero_point = output_details[0]["quantization"]

mffc_avg = np.array([446.02563, 36.123276, -7.691543, 14.370121, -0.68318045, 5.825986, 3.0853918, -0.99041206, 4.941904, 5.33153, 4.3454432, 5.936014, 2.1101217])
mffc_std = np.array([91.35619, 29.026442, 20.318783, 12.534469, 10.299112, 8.434362, 7.7773743, 7.7836747, 8.236834, 7.762929, 7.3308835, 7.83173, 7.062603])

stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=num_frames)

while True:
    audio_data = stream.read(num_frames)
    audio_array = librosa.util.buf_to_float(audio_data)*32768
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, center=False)
    mfcc = ((mfcc.T - mffc_avg) / mffc_std).T

    input_data = mfcc.reshape(input_shape)
    input_data = ( input_data / input_scale + input_zero_point ).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = (output_data.astype(float) - output_zero_point) * output_scale
    
    print(np.argmax(output_data))
