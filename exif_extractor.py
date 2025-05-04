from PIL import Image
from PIL.ExifTags import TAGS
import warnings


def extract_camera_intrinsics(image_path, default_fx=1000, default_fy=1000):
    """
    Extract camera intrinsics (fx, fy, cx, cy) from image metadata.
    If unavailable, return default parameters.

    Args:
        image_path (str): Path to image.
        default_fx (float): Fallback focal length in pixels.
        default_fy (float): Fallback focal length in pixels.

    Returns:
        dict: {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if exif_data is None:
            warnings.warn("No EXIF metadata found, using defaults.")
            raise ValueError
        exif = {TAGS.get(tag, tag): val for tag, val in exif_data.items()}

        focal_length = exif.get('FocalLength')
        if isinstance(focal_length, tuple):  # rational number
            focal_length = focal_length[0] / focal_length[1]

        sensor_width_mm = 36.0  # default full-frame sensor width
        if 'Model' in exif:
            model = exif['Model']
            # Optional: map model → sensor size lookup
            # For demo, we just print:
            print(f"Camera model: {model} (assuming {sensor_width_mm}mm sensor width)")

        width, height = img.size
        cx = width / 2
        cy = height / 2

        if focal_length is not None:
            fx = (focal_length * width) / sensor_width_mm
            fy = (focal_length * height) / sensor_width_mm
            print(f"Extracted focal length: {focal_length}mm → fx={fx:.2f}, fy={fy:.2f}")
        else:
            warnings.warn("Focal length not found, using defaults.")
            fx, fy = default_fx, default_fy

    except Exception as e:
        warnings.warn(f"Metadata extraction failed: {e}. Using defaults.")
        width, height = 1920, 1080  # fallback image size
        fx, fy = default_fx, default_fy
        cx, cy = width / 2, height / 2

    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

intrinsics = extract_camera_intrinsics("depth_maps/example.jpg")
fx = intrinsics['fx']
fy = intrinsics['fy']
cx = intrinsics['cx']
cy = intrinsics['cy']

print(intrinsics)
