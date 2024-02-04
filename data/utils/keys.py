from enum import Enum, auto, IntEnum

class DataSetType(IntEnum):
    DSEC = auto()
    MULTIFLOW2D = auto()

class DataLoading(Enum):
    FLOW = auto()
    FLOW_TIMESTAMPS = auto()
    FLOW_VALID = auto()
    FILE_INDEX = auto()
    EV_REPR = auto()
    BIN_META = auto()
    IMG = auto()
    IMG_TIMESTAMPS = auto()
    DATASET_TYPE = auto()