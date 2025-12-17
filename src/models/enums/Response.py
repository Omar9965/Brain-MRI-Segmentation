from enum import Enum

class Response(Enum):
    max_file_size = 524288000  # 500 MB in bytes
    allowed_types = ["jpg", "png", "tiff", "jpeg", "tif"]
    File_type_not_supported = f"File type is not supported. Types allowed are {', '.join(allowed_types)}"
    File_too_large = f"File is too large. Max file size is {max_file_size // (1024 * 1024)}MB"
    File_Uploaded_Successfully = "File was Uploaded Successfully"
    File_Upload_Failed = "File Upload Failed"