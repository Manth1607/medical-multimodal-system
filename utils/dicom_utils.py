import pydicom
from PIL import Image
import numpy as np

def load_dicom_as_image(file_bytes: bytes) -> Image.Image:
    """Reads DICOM bytes, normalizes pixels, and returns a PIL Image."""
    import io
    dicom_file = pydicom.dcmread(io.BytesIO(file_bytes))
    
    # Get pixel array
    pixel_array = dicom_file.pixel_array
    
    # Normalize to 0-255
    image_2d = pixel_array.astype(float)
    
    # Apply Modality LUT if available (Rescale Intercept/Slope)
    if hasattr(dicom_file, 'RescaleSlope') and hasattr(dicom_file, 'RescaleIntercept'):
        slope = float(dicom_file.RescaleSlope)
        intercept = float(dicom_file.RescaleIntercept)
        image_2d = image_2d * slope + intercept

    # Normalize to 8-bit (0-255)
    max_val = image_2d.max()
    if max_val > 0:
        image_2d = (np.maximum(image_2d, 0) / max_val) * 255.0
    else:
        image_2d = np.zeros_like(image_2d)
    image_2d = np.uint8(image_2d)
    
    # Convert to PIL
    return Image.fromarray(image_2d)
