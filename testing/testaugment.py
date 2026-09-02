import matplotlib.pyplot as plt
import numpy as np
from dataset import PatchDataset

# 1. Initialize your dataset with augmentation turned ON
# (Replace 'path/to/your/data' with your actual data directory)
dataset = PatchDataset("./dataset", augment=True, stain_normalize=True)

# 2. Pick a single image index to test
idx_to_test = 1000 

# 3. Fetch and plot the SAME image 5 times
plt.figure(figsize=(15, 3))
for i in range(5):
    # Fetch the image tensor and label
    img_tensor, label = dataset[idx_to_test]
    
    # 4. Convert the PyTorch tensor back to a standard image for plotting
    # The dataset scales pixels to [-1, 1] and changes shape to ChannelsxHeightxWidth (CHW).
    # We must reverse this to [0, 255] and HeightxWidthxChannels (HWC).
    img_numpy = img_tensor.permute(1, 2, 0).numpy()
    img_numpy = ((img_numpy + 1.0) * 127.5).astype(np.uint8)
    
    # 5. Plot
    plt.subplot(1, 5, i + 1)
    plt.imshow(img_numpy)
    plt.title(f"Augmented View {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()