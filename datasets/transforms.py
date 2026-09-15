"""
Realistic Road Condition Data Augmentations for PotholeGuard-AI.
Simulates:
- Daylight / low light / shadows
- Glare and wet road reflections
- Rain streaks
- Motion blur / camera vibration
- Geometric flips / affine transforms
- CutMix for multi-pothole scenario regularisation
"""
import random
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF

class RoadAugmentation:
    def __init__(self, is_train: bool = True, img_size: int = 512):
        self.is_train = is_train
        self.img_size = img_size

    def add_shadow(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        top_x = random.randint(0, w)
        bot_x = random.randint(0, w)
        shadow_mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array([[(top_x, 0), (w, 0), (w, h), (bot_x, h)]], dtype=np.int32)
        cv2.fillPoly(shadow_mask, poly, 255)
        
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        shadow_factor = random.uniform(0.4, 0.7)
        hls[:, :, 1][shadow_mask == 255] = np.clip(hls[:, :, 1][shadow_mask == 255] * shadow_factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def add_rain(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        rain_layer = np.zeros((h, w), dtype=np.uint8)
        num_drops = random.randint(300, 800)
        for _ in range(num_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 20)
            length = random.randint(10, 25)
            cv2.line(rain_layer, (x, y), (x + random.randint(-3, 3), y + length), 200, 1)
        rain_layer = cv2.blur(rain_layer, (3, 3))
        res = image.copy().astype(np.float32)
        res[:, :, 0] += rain_layer * 0.4
        res[:, :, 1] += rain_layer * 0.4
        res[:, :, 2] += rain_layer * 0.5
        return np.clip(res, 0, 255).astype(np.uint8)

    def add_glare(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(0, int(h * 0.5))
        radius = random.randint(40, 120)
        glare_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(glare_mask, (cx, cy), radius, 1.0, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (101, 101), 0)
        res = image.astype(np.float32) + glare_mask[:, :, None] * random.uniform(80, 160)
        return np.clip(res, 0, 255).astype(np.uint8)

    def __call__(self, image: np.ndarray, mask: np.ndarray, depth: np.ndarray = None):
        """
        Args:
            image: RGB image [H, W, 3] uint8
            mask: Binary mask [H, W] uint8 (0 or 255)
            depth: Depth map [H, W] float32 in [0, 1]
        """
        # Resize to target size
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if depth is not None:
            depth = cv2.resize(depth, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
                if depth is not None:
                    depth = cv2.flip(depth, 1)

            # Random Photometric Augmentations
            if random.random() < 0.3:
                image = self.add_shadow(image)
            if random.random() < 0.2:
                image = self.add_rain(image)
            if random.random() < 0.2:
                image = self.add_glare(image)

            # Brightness & Contrast
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-30, 30)
            image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

            # Random Blur / Camera Shake
            if random.random() < 0.2:
                k = random.choice([3, 5])
                image = cv2.GaussianBlur(image, (k, k), 0)

        # Convert to Tensors
        img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        # Standard ImageNet normalization
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        if depth is not None:
            depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)
        else:
            # Fallback pseudo-depth proxy
            depth_t = torch.zeros_like(mask_t)

        return img_t, mask_t, depth_t
