from butao.env import TaoEnv, get_root_dir, get_root_config
from butao.data import TaoData
from butao.model import TaoModel


def test_get_root_config():
    """Test the get_root_config function."""
    root_config = get_root_config(config="detectnet_v2")
    assert root_config.endswith("butao/configs/detectnet_v2.yaml")


def test_env():
    """Test the set_environment function."""
    env = TaoEnv("tests/test_config.yml")
    root_dir = get_root_dir()

    assert env.params["TAO_MODEL_NAME"] == "detectnet_v2"
    assert env.params["NUM_GPUS"] == 2

    assert env.local_project_dir == f"{root_dir}/notebooks/detectnet_v2"
    assert env.local_data_dir == f"{root_dir}/notebooks/detectnet_v2/data"
    assert env.local_experiment_dir == f"{root_dir}/notebooks/detectnet_v2/detectnet_v2"
    assert env.local_specs_dir == f"{root_dir}/notebooks/detectnet_v2/specs"

    assert env.data_download_dir == "/workspace/tao-experiments/data"
    assert env.user_experiment_dir == "/workspace/tao-experiments/detectnet_v2"
    assert env.specs_dir == "/workspace/tao-experiments/detectnet_v2/specs"


def test_data():
    """Test the print_downloaded_data_info function."""
    data = TaoData("tests/test_config.yml")
    assert data is not None


def test_model():
    """Test the print_downloaded_model_info function."""
    model = TaoModel("tests/test_config.yml")
    assert model is not None
