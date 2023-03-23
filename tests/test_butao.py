import os
from butao import env
from butao.data import TaoData
from butao.model import TaoModel


def test_set_environment():
    """Test the set_environment function."""
    root_dir = env.get_root_dir()
    env.set_environment("tests/test_config.yml", root_dir=root_dir)

    assert os.environ["TAO_MODEL_NAME"] == "detectnet_v2"
    assert os.environ["KEY"] == "tlt_encode"
    assert os.environ["NUM_GPUS"] == "2"

    assert os.environ["LOCAL_PROJECT_DIR"] == f"{root_dir}/notebooks/detectnet_v2"
    assert os.environ["LOCAL_DATA_DIR"] == f"{root_dir}/notebooks/detectnet_v2/data"
    assert (
        os.environ["LOCAL_EXPERIMENT_DIR"]
        == f"{root_dir}/notebooks/detectnet_v2/detectnet_v2"
    )
    assert os.environ["LOCAL_SPECS_DIR"] == f"{root_dir}/notebooks/detectnet_v2/specs"

    assert os.environ["DATA_DOWNLOAD_DIR"] == "/workspace/tao-experiments/data"
    assert (
        os.environ["USER_EXPERIMENT_DIR"] == "/workspace/tao-experiments/detectnet_v2"
    )
    assert os.environ["SPECS_DIR"] == "/workspace/tao-experiments/detectnet_v2/specs"


def test_get_root_config():
    """Test the get_root_config function."""
    root_config = env.get_root_config(config="detectnet_v2")
    assert root_config.endswith("butao/configs/detectnet_v2.yaml")


def test_data():
    """Test the print_downloaded_data_info function."""
    data = TaoData("tests/test_config.yml")
    assert data is not None


def test_model():
    """Test the print_downloaded_model_info function."""
    model = TaoModel("tests/test_config.yml")
    assert model is not None
