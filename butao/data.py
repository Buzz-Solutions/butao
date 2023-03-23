import os
import shutil
import yaml
import zipfile
from pathlib import Path
import urllib.request


class TaoData:
    def __init__(self, config_yaml_path):
        self.local_data_dir = os.environ["LOCAL_DATA_DIR"]
        self.local_specs_dir = os.environ["LOCAL_SPECS_DIR"]

        self.data_download_dir = os.environ["DATA_DOWNLOAD_DIR"]
        self.specs_dir = os.environ["SPECS_DIR"]

        os.makedirs(self.local_data_dir, exist_ok=True)

        # load the config vars
        with open(config_yaml_path, "r") as f:
            env_vars = yaml.safe_load(f)

        self.url_images = env_vars["URL_IMAGES"]
        self.url_labels = env_vars["URL_LABELS"]

        self.spec_file = os.path.join(self.specs_dir, env_vars["DATA_SPEC_FILE"])
        self.local_spec_file = os.path.join(
            self.local_specs_dir, env_vars["DATA_SPEC_FILE"]
        )

        kitti = env_vars["KITTI_FORMAT"]

        # set more of the local and docker folder/file paths
        self.url_labels_filepath = os.path.join(
            self.local_data_dir, Path(self.url_labels).name
        )
        self.url_images_filepath = os.path.join(
            self.local_data_dir, Path(self.url_images).name
        )
        self.training_image_dir = os.path.join(self.local_data_dir, "training/image_2")
        self.training_label_dir = os.path.join(self.local_data_dir, "training/label_2")
        self.testing_image_dir = os.path.join(self.local_data_dir, "testing/image_2")

        if kitti:
            self.local_tfrecords_dir = os.path.join(
                self.local_data_dir, "tfrecords/kitti_trainval"
            )
            self.tfrecords_dir = os.path.join(
                self.data_download_dir, "tfrecords/kitti_trainval/kitti_trainval"
            )
        else:
            self.local_tfrecords_dir = os.path.join(
                self.local_data_dir, "tfrecords/coco_trainval"
            )
            self.tfrecords_dir = os.path.join(
                self.data_download_dir, "tfrecords/coco_trainval/coco_trainval"
            )

        # Create a new directory for the output tfrecords dump.
        if not os.path.exists(self.local_tfrecords_dir):
            os.makedirs(self.local_tfrecords_dir)
        else:
            shutil.rmtree(self.local_tfrecords_dir)
            os.makedirs(self.local_tfrecords_dir)

        print("\nTFrecords conversion spec file for training:")
        with open(self.local_spec_file, "r") as file:
            print(file.read())

    def print_downloaded_data_info(self):
        """Print the number of images and labels in the downloaded data."""
        num_training_images = len(os.listdir(self.training_image_dir))
        num_training_labels = len(os.listdir(self.training_label_dir))
        num_testing_images = len(os.listdir(self.testing_image_dir))

        print(f"Number of images in the train/val set: {num_training_images}")
        print(f"Number of labels in the train/val set: {num_training_labels}")
        print(f"Number of images in the test set: {num_testing_images}")

        # print the first label
        first_label_fn = os.listdir(self.training_label_dir)[0]
        first_label_fp = os.path.join(self.training_label_dir, first_label_fn)
        print(f"First label in the train/val set: \n{first_label_fp}")
        with open(first_label_fp, "r") as file:
            print(file.read())

    def download(self):
        """Download the configured images and labels files"""
        if not os.path.isfile(self.url_images_filepath):
            urllib.request.urlretrieve(self.url_images, self.url_images_filepath)
            print("image archive downloaded")
        else:
            print("image archive already downloaded")

        if not os.path.isfile(self.url_labels_filepath):
            urllib.request.urlretrieve(self.url_labels, self.url_labels_filepath)
            print("label archive downloaded")
        else:
            print("label archive already downloaded")

        print("Extracting the image archive...")
        with zipfile.ZipFile(self.url_images_filepath, "r") as zip_ref:
            zip_ref.extractall(self.local_data_dir)

        print("Extracting the labels archive...")
        with zipfile.ZipFile(self.url_labels_filepath, "r") as zip_ref:
            zip_ref.extractall(self.local_data_dir)

        self.print_downloaded_data_info()
