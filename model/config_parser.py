"""Load and validate the four YAML configs."""
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"


def load_yaml(name: str) -> dict:
    """Read configs/<name>.yaml (name without extension)."""
    # TODO
    raise NotImplementedError


def load_config() -> dict:
    """Return {'data': ..., 'model': ..., 'train': ..., 'report': ...}.

    Validate cross-file consistency, e.g.
      model.spatial_cnn.in_channels == data.patch.channels_C
      model.temporal_lstm.input_size == len(data.columns.temporal_features)
    """
    # TODO
    raise NotImplementedError
