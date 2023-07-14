import numpy as np
import wholeslidedata as wsd

from PIL import Image
from pathlib import Path
from typing import Optional


Image.MAX_IMAGE_PIXELS = 933120000


class WholeSlideImage(object):
    def __init__(
        self, path: Path, spacing: Optional[float] = None, backend: str = "asap"
    ):
        """
        Args:
            path (Path): fullpath to WSI file
        """

        self.path = path
        self.name = path.stem
        self.fmt = path.suffix
        self.wsi = wsd.WholeSlideImage(path, backend=backend)
        self.level_dimensions = self.wsi.shapes
        self.level_downsamples = self.get_downsamples()
        self.spacing = spacing
        self.spacings = self.get_spacings()
        self.backend = backend

        self.spacing_mapping = {a: b for a, b in zip(self.spacings, self.wsi.spacings)}

        self.contours_tissue = None
        self.contours_tumor = None

    def get_downsamples(self):
        level_downsamples = []
        dim_0 = self.level_dimensions[0]
        for dim in self.level_dimensions:
            level_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(level_downsample)
        return level_downsamples

    def get_spacings(self):
        if self.spacing is None:
            common_spacings = [
                0.25,
                0.5,
                1.0,
                2.0,
                4.0,
                8.0,
                16.0,
                32.0,
                64.0,
                128.0,
                256.0,
                512.0,
                1024.0,
            ]
            spacings = [
                common_spacings[np.argmin([abs(cs - s) for cs in common_spacings])]
                for s in self.wsi.spacings
            ]
        else:
            spacings = [
                self.spacing * s / self.wsi.spacings[0] for s in self.wsi.spacings
            ]
        return spacings

    def get_level_spacing(self, level: int = 0):
        return self.spacings[level]

    def get_best_level_for_spacing(
        self, target_spacing: float, ignore_warning: bool = False
    ):
        spacing = self.get_level_spacing(0)
        downsample = target_spacing / spacing
        level, above_tol = self.get_best_level_for_downsample_custom(
            downsample, return_tol_status=True
        )
        if above_tol and not ignore_warning:
            print(
                f"WARNING! The closest natural spacing to the target spacing was more than 15% appart."
            )
        return level

    def get_best_level_for_downsample_custom(
        self, downsample, tol: float = 0.2, return_tol_status: bool = False
    ):
        level = int(np.argmin([abs(x - downsample) for x, _ in self.level_downsamples]))
        above_tol = abs(self.level_downsamples[level][0] / downsample - 1) > tol
        if return_tol_status:
            return level, above_tol
        else:
            return level
