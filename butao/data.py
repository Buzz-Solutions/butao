import os
import shutil
import yaml
import zipfile
from pathlib import Path
import urllib.request


class TaoData:
    def __init__(self, config_yaml_path):
        self.local_data_dir = os.environ["LOCAL_DATA_DIR"]
        self.data_download_dir = os.environ["DATA_DOWNLOAD_DIR"]
        self.local_specs_dir = os.environ["LOCAL_SPECS_DIR"]

        os.makedirs(self.local_data_dir, exist_ok=True)

        # load the config vars
        with open(config_yaml_path, "r") as f:
            env_vars = yaml.safe_load(f)

        self.url_images = env_vars["URL_IMAGES"]
        self.url_labels = env_vars["URL_LABELS"]
        self.data_spec_file = os.path.join(
            self.local_specs_dir, env_vars["DATA_SPEC_FILE"]
        )

    def print_downloaded_data_info(self):
        # print the number of images and labels
        training_image_dir = os.path.join(self.local_data_dir, "training/image_2")
        training_label_dir = os.path.join(self.local_data_dir, "training/label_2")
        testing_image_dir = os.path.join(self.local_data_dir, "testing/image_2")

        num_training_images = len(os.listdir(training_image_dir))
        num_training_labels = len(os.listdir(training_label_dir))
        num_testing_images = len(os.listdir(testing_image_dir))

        print(f"Number of images in the train/val set: {num_training_images}")
        print(f"Number of labels in the train/val set: {num_training_labels}")
        print(f"Number of images in the test set: {num_testing_images}")

        # print the first label
        first_label_file = os.listdir(training_label_dir)[0]
        print(f"First label in the train/val set: {first_label_file}")
        with open(first_label_file, "r") as file:
            print(file.read())

    def download(self):
        """Download the configured images and labels files"""

        url_images_filename = Path(self.url_images).name
        url_labels_filename = Path(self.url_labels).name

        # download the data
        if not os.path.isfile(os.path.join(self.local_data_dir, url_images_filename)):
            urllib.request.urlretrieve(
                self.url_images, os.path.join(self.local_data_dir, url_images_filename)
            )
            print("image archive downloaded")
        else:
            print("image archive already downloaded")

        if not os.path.isfile(os.path.join(self.local_data_dir, url_labels_filename)):
            urllib.request.urlretrieve(
                self.url_labels, os.path.join(self.local_data_dir, url_labels_filename)
            )
            print("label archive downloaded")
        else:
            print("label archive already downloaded")

        # unzip the data
        with zipfile.ZipFile(
            f"{self.local_data_dir}/{url_images_filename}.zip", "r"
        ) as zip_ref:
            zip_ref.extractall(f"{self.local_data_dir}")

        with zipfile.ZipFile(
            f"{self.local_data_dir}/{url_labels_filename}.zip", "r"
        ) as zip_ref:
            zip_ref.extractall(f"{self.local_data_dir}")

        self.print_downloaded_data_info()

    def convert(self, kitti=True):
        """Convert the configured images and labels files to tfrecords."""
        if kitti:
            tfrecords_dir = os.path.join(
                self.local_data_dir, "tfrecords/kitti_trainval"
            )
            docker_tfrecords_dir = os.path.join(
                self.data_download_dir, "tfrecords/kitti_trainval/kitti_trainval"
            )
            print("Converting kitti data to Tfrecords for trainval dataset")
        else:
            tfrecords_dir = os.path.join(self.local_data_dir, "tfrecords/coco_trainval")
            docker_tfrecords_dir = os.path.join(
                self.data_download_dir, "tfrecords/coco_trainval/coco_trainval"
            )
            print("Converting coco data to Tfrecords for trainval dataset")

        print("TFrecords conversion spec file for training")
        with open(self.data_spec_file, "r") as file:
            print(file.read())

        # Creating a new directory for the output tfrecords dump.
        print("Converting Tfrecords for trainval dataset")
        if not os.path.exists(tfrecords_dir):
            os.makedirs(tfrecords_dir)
        else:
            shutil.rmtree(tfrecords_dir)
            os.makedirs(tfrecords_dir)

        exec_cmd = f"tao detectnet_v2 dataset_convert \
        -d {self.data_spec_file} -o {docker_tfrecords_dir}"
        print("executing command: \n", exec_cmd)
        os.system(exec_cmd)
