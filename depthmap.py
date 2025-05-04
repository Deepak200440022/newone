from transformers import pipeline
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# Initialize the depth estimation pipeline
pipe = pipeline(task="depth-estimation", model="Intel/dpt-large")

# Load image from URL
url ="/home/deepak/Downloads/morskie-oko-tatry.jpg"
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(url)

# Run depth estimation
result = pipe(image)

# Get depth map (PIL Image)
depth_map = result["depth"]

# Save depth map as PNG
depth_map.save("depth_map.png")
print("Depth map saved as depth_map.png")

# Optional: display depth map inline with matplotlib
plt.imshow(depth_map, cmap="gray")
plt.title("Predicted Depth Map")
plt.axis("off")
plt.show()
