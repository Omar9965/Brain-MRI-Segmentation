from .BaseController import BaseController
from models import Response

class DataController(BaseController):
    async def validate_images(self, images):
        for image in images:
            if image.filename.split('.')[-1].lower() not in Response.allowed_types.value:
                return False, Response.File_type_not_supported.value
            
            # Read file to get size
            contents = await image.read()
            size = len(contents)
            
            # Reset file pointer to beginning
            await image.seek(0)
            
            if size > Response.max_file_size.value:
                return False, Response.File_too_large.value
        
        return True, Response.File_Uploaded_Successfully.value