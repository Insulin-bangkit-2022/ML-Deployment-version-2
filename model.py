import tensorflow as tf
import io
import numpy as np


def predict(userData):
    model = tf.keras.models.load_model('mymodel.h5')
    model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

    pred = np.array(userData[40, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1])
    prediction = model.predict(pred)
    #print (np.shape(pred))
    #output = prediction([])
    #y_pred = model.predict(x_test)
    for value in prediction :
        if value > 0.5:
            value = 1
        else:
            value = 0
    return value
    
if __name__ == '__main__':
	print(predict(np.array([40, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1], dtype=np.float32)))

    #print(output)
    
    #print (prediction)
    # label = np.where(classes[0] > 0.5, 1,0)"""
    