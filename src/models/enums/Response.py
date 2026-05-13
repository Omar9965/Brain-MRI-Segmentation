from enum import Enum

# Define constants outside the enum so f-strings reference actual values, not enum members
_MAX_FILE_SIZE = 524288000  # 500 MB in bytes
_ALLOWED_TYPES = ["jpg", "png", "tiff", "jpeg", "tif"]


class Response(Enum):
    max_file_size = _MAX_FILE_SIZE
    allowed_types = _ALLOWED_TYPES
    File_type_not_supported = f"File type is not supported. Types allowed are {', '.join(_ALLOWED_TYPES)}"
    File_too_large = f"File is too large. Max file size is {_MAX_FILE_SIZE // (1024 * 1024)}MB"
    File_Uploaded_Successfully = "File was Uploaded Successfully"
    File_Upload_Failed = "File Upload Failed"