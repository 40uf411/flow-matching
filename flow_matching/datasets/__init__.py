from flow_matching.datasets.synthetic_datasets import (
    DatasetCheckerboard,
    DatasetInvertocat,
    DatasetMixture,
    DatasetMoons,
    DatasetSiggraph,
    SyntheticDataset,
)

TOY_DATASETS: dict[str, type[SyntheticDataset]] = {
    "moons": DatasetMoons,
    "mixture": DatasetMixture,
    "siggraph": DatasetSiggraph,
    "checkerboard": DatasetCheckerboard,
    "invertocat": DatasetInvertocat,
}
