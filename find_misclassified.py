import tensorflow as tf
import numpy as np

# Load dataset
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_test = x_test / 255.0
x_test = np.expand_dims(x_test, -1)   # shape (10000, 28, 28, 1)

# Load your partial model
model = tf.keras.models.load_model("notMNIST-Partial.keras")

# Get predictions
pred_prob = model.predict(x_test, verbose=0)
pred_labels = np.argmax(pred_prob, axis=1)

# Find misclassified indices
misclassified = np.where(pred_labels != y_test)[0]

print(f"Found {len(misclassified)} misclassified images.")
print("Here are the first 30 (index, true_label, predicted_label):\n")

for idx in misclassified[:100]:
    print(f"Index {idx}  True: {y_test[idx]}  Predicted: {pred_labels[idx]}")

'''
Found 689 misclassified images.
Here are the first 30 (index, true_label, predicted_label):

Index 1  True: 0  Predicted: 6
Index 8  True: 3  Predicted: 8
Index 21  True: 2  Predicted: 4
Index 23  True: 5  Predicted: 8
Index 28  True: 1  Predicted: 6
Index 61  True: 0  Predicted: 1
Index 65  True: 6  Predicted: 2
Index 85  True: 2  Predicted: 8
Index 93  True: 3  Predicted: 0
Index 98  True: 5  Predicted: 7
Index 102  True: 7  Predicted: 0
Index 105  True: 2  Predicted: 6
Index 108  True: 3  Predicted: 6
Index 138  True: 9  Predicted: 6
Index 148  True: 7  Predicted: 1
Index 155  True: 3  Predicted: 0
Index 158  True: 3  Predicted: 1
Index 161  True: 1  Predicted: 6
Index 167  True: 4  Predicted: 6
Index 187  True: 6  Predicted: 8
Index 198  True: 3  Predicted: 9
Index 240  True: 1  Predicted: 6
Index 242  True: 6  Predicted: 2
Index 247  True: 8  Predicted: 2
Index 263  True: 5  Predicted: 9
Index 264  True: 8  Predicted: 3
Index 281  True: 3  Predicted: 0
Index 320  True: 6  Predicted: 8
Index 325  True: 8  Predicted: 0
Index 346  True: 2  Predicted: 4
Index 361  True: 6  Predicted: 5
Index 373  True: 8  Predicted: 9
Index 396  True: 2  Predicted: 6
Index 436  True: 9  Predicted: 3
Index 446  True: 4  Predicted: 2
Index 453  True: 5  Predicted: 4
Index 456  True: 7  Predicted: 5
Index 481  True: 7  Predicted: 0
Index 483  True: 4  Predicted: 6
Index 494  True: 9  Predicted: 1
Index 512  True: 8  Predicted: 0
Index 515  True: 2  Predicted: 8
Index 551  True: 5  Predicted: 4
Index 555  True: 3  Predicted: 8
Index 615  True: 2  Predicted: 8
Index 618  True: 7  Predicted: 8
Index 635  True: 6  Predicted: 4
Index 653  True: 3  Predicted: 7
Index 667  True: 5  Predicted: 6
Index 685  True: 7  Predicted: 6
Index 688  True: 5  Predicted: 6
Index 709  True: 5  Predicted: 0
Index 722  True: 4  Predicted: 1
Index 743  True: 2  Predicted: 1
Index 767  True: 6  Predicted: 2
Index 774  True: 9  Predicted: 6
Index 812  True: 7  Predicted: 8
Index 816  True: 8  Predicted: 1
Index 824  True: 7  Predicted: 1
Index 827  True: 8  Predicted: 9
Index 829  True: 5  Predicted: 4
Index 835  True: 2  Predicted: 4
Index 839  True: 9  Predicted: 1
Index 850  True: 1  Predicted: 8
Index 882  True: 0  Predicted: 7
Index 887  True: 9  Predicted: 3
Index 913  True: 1  Predicted: 7
Index 918  True: 3  Predicted: 1
Index 921  True: 6  Predicted: 8
Index 923  True: 3  Predicted: 9
Index 933  True: 5  Predicted: 8
Index 973  True: 4  Predicted: 1
Index 1005  True: 8  Predicted: 9
Index 1014  True: 8  Predicted: 6
Index 1022  True: 4  Predicted: 6
Index 1026  True: 2  Predicted: 6
Index 1031  True: 3  Predicted: 4
Index 1052  True: 8  Predicted: 9
Index 1066  True: 2  Predicted: 6
Index 1072  True: 1  Predicted: 7
Index 1075  True: 9  Predicted: 8
Index 1084  True: 2  Predicted: 6
Index 1104  True: 1  Predicted: 9
Index 1106  True: 5  Predicted: 9
Index 1108  True: 7  Predicted: 5
Index 1117  True: 4  Predicted: 8
Index 1125  True: 8  Predicted: 9
Index 1173  True: 2  Predicted: 6
Index 1179  True: 8  Predicted: 1
Index 1181  True: 5  Predicted: 1
Index 1184  True: 8  Predicted: 5
Index 1196  True: 8  Predicted: 9
Index 1201  True: 2  Predicted: 4
Index 1202  True: 3  Predicted: 8
Index 1204  True: 6  Predicted: 4
Index 1208  True: 6  Predicted: 2
Index 1224  True: 1  Predicted: 3
Index 1239  True: 4  Predicted: 6
Index 1249  True: 8  Predicted: 7
Index 1253  True: 4  Predicted: 1
'''