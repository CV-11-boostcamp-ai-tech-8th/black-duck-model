import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from models.network_swinir import SwinIR


def pad_to_window_size(x, window_size=8):
    """
    Pad image to be divisible by window_size for SwinIR processing
    """
    _, _, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    return x, pad_h, pad_w

def unpad(x, pad_h, pad_w):
    """
    Remove padding from output tensor
    """
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]

    return x


def main():
    input_dir = "./datasets/plate_data/test"
    output_dir = "./datasets/plate_data/test_sr"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize SwinIR model for 4x super-resolution
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=240,
        num_heads=[8] * 9,
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="3conv",
    )

    # Setup device and load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load pre-trained weights (EMA parameters)
    checkpoint_path = "./weights/SwinIR_L.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["params_ema"], strict=True)

    print(f"Model loaded on {device}")
    print(f"Processing images from: {input_dir}\n")

    
    for fname in sorted(os.listdir(input_dir)):
        # Skip non-image files
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        
        in_path = os.path.join(input_dir, fname)
        name, ext = os.path.splitext(fname)
        out_path = os.path.join(output_dir, f"{name}_sr{ext}")

        img = cv2.imread(in_path)
        if img is None:
            print(f"Failed to load: {fname}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # Pad image to window size
        img_padded, pad_h, pad_w = pad_to_window_size(img, window_size=8)

        # Run super-resolution inference
        with torch.no_grad():
            sr_output = model(img_padded)

        # Remove padding
        sr_output = unpad(sr_output, pad_h * 4, pad_w * 4)  # x4 upscale

        sr_img = sr_output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        sr_img = (sr_img * 255.0).astype(np.uint8)
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(out_path, sr_img)
        print(f"{fname} → SR completed")

    print("\nSuper-resolution completed")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()

