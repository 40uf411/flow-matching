from flow_matching.datasets.synthetic_datasets import (
    DatasetCheckerboard,
    DatasetInvertocat,
    DatasetMixture,
    DatasetMoons,
    DatasetSiggraph,
    SyntheticDataset,
)
from flow_matching.datasets.volume_datasets import (
    BinaryNpyVolumeDataset,
    get_volume_dataset,
    get_volume_transform,
)

TOY_DATASETS: dict[str, type[SyntheticDataset]] = {
    "moons": DatasetMoons,
    "mixture": DatasetMixture,
    "siggraph": DatasetSiggraph,
    "checkerboard": DatasetCheckerboard,
    "invertocat": DatasetInvertocat,
}
