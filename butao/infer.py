import os
import matplotlib.pyplot as plt
from PIL import Image

from butao.env import TaoEnv


valid_image_ext = [".jpg", ".png", ".jpeg", ".ppm"]


class TaoInfer(TaoEnv):
    def __init__(self, config_yaml_path):
        TaoEnv.__init__(self, config_yaml_path)

        self.inference_spec = os.path.join(
            self.specs_dir, self.params["INFERENCE_SPEC_FILE"]
        )

        self.sample_data_dir = os.path.join(self.data_download_dir, "test_samples")
        self.local_sample_data_dir = os.path.join(self.local_data_dir, "test_samples")

        self.infer_dir = os.path.join(self.user_experiment_dir, "tlt_infer_testing")

        self.local_annotated_dir = os.path.join(
            self.local_experiment_dir, "tlt_infer_testing", "images_annotated"
        )
        self.local_predictions_dir = os.path.join(
            self.local_experiment_dir, "tlt_infer_testing", "labels"
        )
        os.makedirs(self.local_sample_data_dir, exist_ok=True)

    def plot_image_grid(self, rows, cols, num_images=10):
        filepaths = [
            os.path.join(self.local_annotated_dir, image)
            for image in os.listdir(self.local_annotated_dir)
            if os.path.splitext(image)[1].lower() in valid_image_ext
        ]

        num_images = len(filepaths) if not num_images < len(filepaths) else num_images
        fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        n = 0
        for i in range(rows):
            for j in range(cols):
                image = Image.open(filepaths[n])
                ax[i][j].imshow(image)
                image.close()
                ax[i][j].axis("off")
                n += 1
                if n >= num_images:
                    break
        plt.tight_layout()
        plt.show()
