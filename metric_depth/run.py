import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
# import open3d as o3d
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import cv2

# Global settings
FL = 3263.5
FY = 256 * 0.6
FX = 256 * 0.6
NYU_DATA = False
FINAL_HEIGHT = 3072
FINAL_WIDTH = 2048
DATASET = 'nyu' # Lets not pick a fight with the model's dataloader

def process_images(model, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    depth_min = 0
    depth_max = 0
    image_paths = glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.jpg'))
    image_paths.sort()
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing Images")):
        try:
            color_image = Image.open(image_path).convert('RGB')
            original_width, original_height = color_image.size
            image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

            pred = model(image_tensor, dataset=DATASET)
            if isinstance(pred, dict):
                pred = pred.get('metric_depth', pred.get('out'))
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]
            pred = pred.squeeze().detach().cpu().numpy()
            resized_pred = Image.fromarray(pred).resize((original_width, original_height), Image.NEAREST)
            np_resized_pred=np.array(resized_pred)
            np.save(f"{output_dir}/{i:06d}.npy", np_resized_pred)
            np_resized_pred=np_resized_pred*1000
            np_resized_pred=np_resized_pred.astype(np.uint16)
            depth_min = depth_min if depth_min < np_resized_pred.min() else np_resized_pred.min()
            depth_max =  depth_max if depth_max > np_resized_pred.max() else np_resized_pred.max()
            # cv2.imwrite(f"{output_dir}/{i:06d}.png", np_resized_pred ) # pred.astype(np.uint16)

            # Resize color image and depth to final size
            resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
            resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

            focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
            x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
            x = (x - FINAL_WIDTH / 2) / focal_length_x
            y = (y - FINAL_HEIGHT / 2) / focal_length_y
            z = np.array(resized_pred)
            # points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            # colors = np.array(resized_color_image).reshape(-1, 3) / 255.0

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".ply"), pcd)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"minimum depth for video: {depth_min}, maximum depth: {depth_max}")


def main(model_name, pretrained_resource, input_dir, output_dir):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    process_images(model, input_dir, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_indoor.pt', help="Pretrained resource to use for fetching weights.")
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    args = parser.parse_args()
    main(args.model, args.pretrained_resource, args.input_dir, args.output_dir)

    