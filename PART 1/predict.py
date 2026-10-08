#Original Author: Jonathan Hudson
#CPSC 433 F24

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pyscreenshot as ImageGrab

def main():
     class_names = check_args()
     print(f"--Load Model {sys.argv[2]}--")
     #Load the model that should be in sys.argv[2]
     model = tf.keras.models.load_model(sys.argv[2])    
     print(f"--Load Image {sys.argv[3]}--")
     img = plt.imread(sys.argv[3])
     if np.amax(img.flatten()) > 1:
          img = img/255
     img = 1 - img
     print(f"--Predict as Class {sys.argv[4]}--")
     predict(model, class_names, img, int(sys.argv[4]))

def predict(model, class_names, img, true_label):
     img = np.array([img])
     img = img.reshape(1, 28, 28, 1)
     prediction = model.predict(img, verbose=0)[0]   # get probability array
     predicted_label = np.argmax(prediction)
     plot(class_names, prediction, true_label, predicted_label, img[0])
     plt.show()

def check_args():
     if(len(sys.argv) == 1):
         print("No arguments so using defaults")
         sys.argv = ["predict.py", "MNIST", "MNIST.keras", "image.png", input("Predict the class of image.png(index):")]
     if(len(sys.argv) != 5):
          print("Usage python predict.py <MNIST,notMNIST> <model.keras> <image.png> <prediction class index>")
          sys.exit(1)
     if sys.argv[1] == "MNIST":
          print("--Dataset MNIST--")
          class_names = list(range(10))
     elif sys.argv[1] == "notMNIST":
          print("--Dataset notMNIST--")
          class_names = ["A","B","C","D","E","F","G","H","I","J"]
     else:
          print(f"Choose MNIST or notMNIST, not {sys.argv[1]}")
          sys.exit(2)
     if sys.argv[2][-6:] != ".keras":
          print(f"{sys.argv[2]} is not a keras extension")
          sys.exit(3)
     if sys.argv[3][-4:] != ".png":
          print(f"{sys.argv[3]} is not a png extension")
          sys.exit(3)
     img = plt.imread(sys.argv[3])
     if len(img.shape) != 2:
          print("Image is not grey scale!")
          sys.exit(4)
     if img.shape != (28,28):
          print("Image is not 28 by 28!")
          sys.exit(4)
     if not sys.argv[4].isdigit():
          print(f"{sys.argv[4]} is not an integer (0-9)")
          sys.exit(3)
     if int(sys.argv[4]) < 0 or int(sys.argv[4]) > 9 :
          print(f"{sys.argv[4]} is not an integer (0-9)")
          sys.exit(3)
     return class_names

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
