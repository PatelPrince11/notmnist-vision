#Original Author: Jonathan Hudson
#CPSC 433 F24

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main():
     class_names, data = check_args()
     x_test, y_test = data
     print(f"--Load Model {sys.argv[2]}--")
     #Load the model that should be in sys.argv[2]
     model = None     
     pick = input(f"Pick test_image (0 -> {len(x_test)-1}):")
     while pick.isdigit() and int(pick) >= 0 and int(pick) < len(x_test):
        pick = int(pick)
        img = x_test[pick]
        guess = y_test[pick]
        print(f"--Should be Class {guess}--")
        predict(model, class_names, img, guess)
        pick = input(f"Pick test_image (0 -> {len(x_test)-1}):")
     print("Done")

def predict(model, class_names, img, true_label):
    img = np.array([img])
    #Replace these two lines with code to make a prediction
    prediction = [1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10]
    #Determine what the predicted label is
    predicted_label = 0
    plot(class_names, prediction, true_label, predicted_label, img[0])
    plt.show()

def check_args():
     if(len(sys.argv) == 1):
        print("No arguments so using defaults")
        if input("Y for MNIST, otherwise notMNIST:") == "Y":
             sys.argv = ["predict_test.py", "MNIST", "MNIST.keras"]
        else:
             sys.argv = ["predict_test.py", "notMNIST", "notMNIST.keras"]
     if(len(sys.argv) != 3):
          print("Usage python predict_test.py <MNIST,notMNIST> <model.keras>")
          sys.exit(1)
     if sys.argv[1] == "MNIST":
          print("--Dataset MNIST--")
          class_names = list(range(10))
          mnist = tf.keras.datasets.mnist
          (x_train, y_train), (x_test, y_test) = mnist.load_data()
          x_train, x_test = x_train / 255.0, x_test / 255.0
          data = (x_test, y_test)
     elif sys.argv[1] == "notMNIST":
          print("--Dataset notMNIST--")
          class_names = ["A","B","C","D","E","F","G","H","I","J"]
          with np.load("notMNIST.npz", allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
          x_train, x_test = x_train / 255.0, x_test / 255.0
          data = (x_test, y_test)
     else:
          print(f"Choose MNIST or notMNIST, not {sys.argv[1]}")
          sys.exit(2)
     if sys.argv[2][-6:] != ".keras":
          print(f"{sys.argv[2]} is not a keras extension")
          sys.exit(3)
     return class_names, data

def plot(class_names, prediction, true_label, predicted_label, img):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(prediction),class_names[true_label]),color=color)
    plt.subplot(1,2,2)
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(class_names, prediction, color="#777777")
    plt.ylim([0, 1])
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

if __name__ == "__main__":
    main()
