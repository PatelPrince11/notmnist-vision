import numpy as np
import tensorflow as tf

with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_test = f['x_test']
    y_test = f['y_test']

x_test = x_test / 255.0
x_test = np.expand_dims(x_test, -1)

# indices_to_check = [1,8,21,23,28,61,65,85,93,98,102,105,108,138,148,155,158,161,
#                     167,187,198,240,242,247,263,264,281,320,325,346,361,373,396,
#                     436,446,453,456,481,483,494,512,515,551,555,615,618,635,653,
#                     667,685,688,709,722,743,767,774,812,816,824,827,829,835,839,
#                     850,882,887,913,918,921,923,933,973,1005,1014,1022,1026,1031,
#                     1052,1066,1072,1075,1084,1104,1106,1108,1117,1125,1173,1179,
#                     1181,1184,1196,1201,1202,1204,1208,1224,1239,1249,1253]

indices_to_check = [148, 446, 1201]

partial_model = tf.keras.models.load_model("notMNIST-Partial.keras")
complete_model = tf.keras.models.load_model("notMNIST-Complete.keras")

for idx in indices_to_check:
    img = x_test[idx].reshape(1,28,28,1)
    true_label = y_test[idx]
    
    pred_partial_probs = partial_model.predict(img, verbose=0)[0]
    pred_complete_probs = complete_model.predict(img, verbose=0)[0]
    
    pred_partial = np.argmax(pred_partial_probs)
    pred_complete = np.argmax(pred_complete_probs)
    
    partial_percentages = [f"{p*100:.2f}%" for p in pred_partial_probs]
    complete_percentages = [f"{p*100:.2f}%" for p in pred_complete_probs]
    
    if pred_partial != true_label and pred_complete == true_label:
        print(f"Index: {idx} | True: {true_label}")
        print(f"Partial Prediction: {pred_partial} | Confidence: {partial_percentages}")
        print(f"Complete Prediction: {pred_complete} | Confidence: {complete_percentages}")