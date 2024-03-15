
# ### ! [General] ! ###
PROJECT_TAG = 'CTSeg_'
SEED = 42
NAS = "//192.168.0.000/XXX/"
DTYPE = "CBCT"   # CT MDCT CBCT
SOURCE_COMPANY = 'YYY'
GPU = False

# ### ! [Inference] ! ###
INFERENCE_EVAL = True
POSTPROCESS = False

PATI_NUM = "00000"

# *Temp
PREDICTED_LOG_DIR = "/Users/dyanlee/workspace/CTSegmentation/Logs"
# *Temp
MODEL_PATH = "checkpoints.pth"

# ### ! [Dataset Loading] ! ###
SOURCE = NAS + "/Data/CT/"
CONVERTED_DIR = SOURCE + DTYPE + "/" + SOURCE_COMPANY + "/vti/"


# ### ! [Training] ! ###
# ### Transfer Learning
TRANSFER = False
PRETRAINED_MODEL = "checkpoints.pth"

# ### Model Hyper-params
EPOCH = 10
LR = 1e-4  # 0.1
IN_DIM, OUT_DIM = 5, 3  # N_CLASS = 3

# ### Loader
LOAD_MODE = "ORI"  # 'NORM' #"ORI"
SLICE_BATCH = 2

VAL_SPLIT = .2
PATI_BATCH = 1
PATI_SHUFFLE = True
NUM_SLICES = 5
SLICE_SHUFFLE = True

# ### preprocessing
AUGMENTATION = False
SCALE = 0.5  # Downsampling == preprocessing for smaller resizing
