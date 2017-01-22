import cv2

# Performs Image Processing into suitable input

def getSample(filename):
    im_gray = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    resized = cv2.resize(im_gray, (6, 6), interpolation = cv2.INTER_AREA)
    _, im_bw = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw

# Converts to GrayScale, then Black and White using Otsu Threshold
# Rescales from 28 x 28 to 6 x 6 image
def getTrainingSamples(n, training=900):
    train = []
    for i in xrange(training):
        filename = 'Samples/data_{0}/{0}_{1}.jpg'.format(n,i)
        im_gray = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        resized = cv2.resize(im_gray, (6, 6), interpolation = cv2.INTER_AREA)
        _, im_bw = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        train.append(im_bw)
    return train

# Converts to GrayScale, then Black and White using Otsu Threshold
# Rescales from 28 x 28 to 6 x 6 image
def getTestSamples(n, test=100):
    testing = []
    for i in xrange(1000-test, 1000):
        filename = 'Samples/data_{0}/{0}_{1}.jpg'.format(n,i)
        im_gray = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        resized = cv2.resize(im_gray, (6, 6), interpolation = cv2.INTER_AREA)
        _, im_bw = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        testing.append(im_bw)
    return testing


def drawImage(filename):
    image = cv2.imread(filename)
    cv2.imshow("Character", image)
