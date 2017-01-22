ACT_THRES = 55.
BIAS = 1

SAMPLE_SIZE = 6 * 6
CHARACTERS = [0,1,2,3,4,5,6,7,8,9]

NUM_LAYERS = 1
INPUT_LAYER = SAMPLE_SIZE
HIDDEN_LAYER = 45
OUTPUT_LAYER = len(CHARACTERS)
DEFAULT_LAYERS = [INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER]

LEARNING_RATE = 5
NUM_TRAIN = 900
NUM_TEST = 100


TRAIN_FLAGS = {'-t': 'numTrainSamples', '-h': 'numHiddenLayers', 
               '-n': 'neuronsPerLayer', '-r': 'learningRate', 
               '-a': 'actThres'       , '-b': 'bias'}

TEST_FLAGS = {'-T': 'numTestSamples'}

SPECIAL_FLAGS = {'-s', '-S'}