"""Dataset manifest: tracks known datasets, their location, sample count,
and format version — so a job references a dataset_id instead of a raw path.
"""

from pathlib import Path

MANIFEST_PATH = Path("data/dataset_manifest.json")


class DatasetRegistry:
    def __init__(self, manifest_path: Path = MANIFEST_PATH):
        self.manifest_path = manifest_path

    def register(self, dataset_id: str, path: str, num_samples: int) -> None:
        # TODO: read-modify-write the manifest JSON
        pass

    def resolve(self, dataset_id: str) -> str:
        # TODO: look up path for dataset_id, raise a clear error if missing
        raise NotImplementedError
