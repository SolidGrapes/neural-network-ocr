Packages/Dependencies:
brew install python2, scipy, numpy, opencv

Training Flags: 
    -t : the number of training samples per digit
    -h : the number of hidden layers in the network
    -n : the number of neurons per hidden layer
    -b : the weight of the bias
    -r : the learning rate of the network, as defined above
    -a : the activation threshold for the sigmoid function

Test Flags:
    -T : the number of test samples per digit
    -s : make predictions for a single image
    -S : make predictions for a single image and depict the black/white Otsu
         Thresholded version of the image in console with {0,255}'s

Example Terminal Commands From Directory Containing Source Files:

python main.py -h 1 -n 45 -t 800 -a 55 -b 1 -r 5 -T 100

python main.py -a 75 -b 3 -t 900

python main.py -n 55 -t 600

python main.py -s Samples/data_0/0_13.jpg

python main.py -S Samples/data_3/3_24.jpg

python main.py -s Samples/data_2/2_12.jpg -b 3 -t 900 -a 75

Flag Notes:
    - There are only 1000 sample images for each digit so assign the
      number of training images and test images accordingly. 
    - Default values are stored in the constants module. 
    - Order of flags does not matter.
    - Bias here is computed assuming (x/a + b) rather than ((x + b)/a), hence the
      smaller bias values.


Relevant Source Modules:
main.py     : This module is responsible for parsing command line flags, the 
              top-level training and testing of our neural network, and 
              determining the cumulative accuracy of our neural network as well 
              as the accuracy per digit. 

getSample   : This module is responsible for reading in images from the data set, 
              resizing them, and thresholding them from grayscale to black and 
              white. Image processing is done via the OpenCV library.

neurons     : This module is responsible for instantiating and maintaining a 
              single neuron in the network. Each neuron stores the weights of 
              its incoming edges and its activation value. Initialization of 
              weights on incoming edges is also done in this module. 

neuralnet   : This module is responsible for instantiating and maintaining the 
              top-level neural network. It handles the logic for initializing 
              each layer and implements the FeedForward and BackPropagate 
              algorithms. In addition, activation values for neurons are 
              computed in this module. 