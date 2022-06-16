import os

PATH_OYSTERS = "Oysters"
PATH_OYSTERS_IMAGES = os.path.sep.join([PATH_OYSTERS, "images"])
PATH_OYSTERS_ANNOTATIONS = os.path.sep.join([PATH_OYSTERS, "annotations"])

PATH_DATASET = "OysterDataset"
PATH_POSITIVE = os.path.sep.join([PATH_DATASET, "Oyster"])
PATH_NEGATIVE = os.path.sep.join([PATH_DATASET, "NoOyster"])

MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

INPUT_DIMS = (512, 512)

MODEL_PATH = "oyster_detector.h5"
ENCODER_PATH = "label_encoder.pickle"

MIN_PROBA = 0.99