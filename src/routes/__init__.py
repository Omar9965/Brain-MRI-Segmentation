from .schemas.DataSchema import SegmentationResult, MultipleSegmentationResponse, BatchSubmissionResponse
from .base import router as base_router
from .data import router as data_router
from .websocket import router as websocket_router