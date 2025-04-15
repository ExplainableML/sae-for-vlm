import os
import shutil
import argparse

def main(imagenet_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for class_folder in sorted(os.listdir(imagenet_root)):
        class_path = os.path.join(imagenet_root, class_folder)

        if os.path.isdir(class_path):
            images = sorted(os.listdir(class_path))
            if images:
                first_image = images[0]
                src_path = os.path.join(class_path, first_image)
                dest_path = os.path.join(output_dir, f"{class_folder}_{first_image}")

                shutil.copy(src_path, dest_path)
                print(f"Copied {first_image} from {class_folder}")

    print("Done selecting first images for each class!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy the first image from each ImageNet class folder.")
    parser.add_argument("--imagenet_root", required=True, help="Path to the ImageNet training dataset root directory")
    parser.add_argument("--output_dir", default="./images_imagenet", help="Output directory for selected images")

    args = parser.parse_args()
    main(args.imagenet_root, args.output_dir)
