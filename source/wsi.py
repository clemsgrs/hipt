import cv2
import numpy as np
import wholeslidedata as wsd

from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional


Image.MAX_IMAGE_PIXELS = 933120000


def find_common_spacings(spacings_1, spacings_2, tolerance: float = 0.05):
    common_spacings = []
    for s1 in spacings_1:
        for s2 in spacings_2:
            # check how far appart these two spacings are
            if abs(s1 - s2) / s1 <= tolerance:
                common_spacings.append((s1, s2))
    return common_spacings


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
            spacings = self.wsi.spacings
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

    def loadSegmentation(
        self,
        mask_fp: Path,
        downsample: int,
    ):

        mask = WholeSlideImage(mask_fp, backend=self.backend)

        # ensure mask and slide have at least one common spacing
        common_spacings = find_common_spacings(self.spacings, mask.spacings, tolerance=0.1)
        assert len(common_spacings) >= 1, f"The provided segmentation mask (spacings={mask.spacings}) has no common spacing with the slide (spacings={self.spacings}). A minimum of 1 common spacing is required."

        seg_level = self.get_best_level_for_downsample_custom(downsample)
        seg_spacing = self.get_level_spacing(seg_level)

        # check if this spacing is present in common spacings
        is_in_common_spacings = seg_spacing in [s for s,_ in common_spacings]
        if not is_in_common_spacings:
            # find spacing that is common to slide and mask and that is the closest to seg_spacing
            closest = np.argmin([abs(seg_spacing-s) for s,_ in common_spacings])
            closest_common_spacing = common_spacings[closest][0]
            seg_spacing = closest_common_spacing
            seg_level = self.get_best_level_for_spacing(seg_spacing)

        m = mask.wsi.get_slide(spacing=seg_spacing)
        m = m[..., 0]

        self.binary_mask = m
        return seg_level, seg_spacing

    def segmentTissue(
        self,
        downsample: int,
        sthresh: int = 20,
        sthresh_up: int = 255,
        mthresh: int = 7,
        close: int = 0,
        use_otsu: bool = False,
    ):
        """
        Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """

        seg_level = self.get_best_level_for_downsample_custom(downsample)
        seg_spacing = self.get_level_spacing(seg_level)
        img = self.wsi.get_slide(spacing=seg_spacing)
        img = np.array(Image.fromarray(img).convert("RGBA"))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring

        # Thresholding
        if use_otsu:
            _, img_thresh = cv2.threshold(
                img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY
            )
        else:
            _, img_thresh = cv2.threshold(
                img_med, sthresh, sthresh_up, cv2.THRESH_BINARY
            )

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        m = img_thresh.astype(int) // 255

        self.binary_mask = m
        return seg_level, seg_spacing

    def detect_contours(
        self, img_thresh, spacing: float, seg_level: int, filter_params: Dict[str, int]
    ):
        def _filter_contours(contours, hierarchy, filter_params):
            """
            Filter contours by: area.
            """
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
            all_holes = []

            # loop through foreground contour indices
            for cont_idx in hierarchy_1:
                # actual contour
                cont = contours[cont_idx]
                # indices of holes contained in this contour (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # take contour area (includes holes)
                a = cv2.contourArea(cont)
                # calculate the contour area of each hole
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                if a == 0:
                    continue
                if a > filter_params["a_t"]:
                    filtered.append(cont_idx)
                    all_holes.append(holes)

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]

            hole_contours = []
            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids]
                unfilered_holes = sorted(
                    unfiltered_holes, key=cv2.contourArea, reverse=True
                )
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[: filter_params["max_n_holes"]]
                filtered_holes = []

                # filter these holes
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params["a_h"]:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours

        spacing_level = self.get_best_level_for_spacing(spacing)
        current_scale = self.level_downsamples[spacing_level]
        target_scale = self.level_downsamples[seg_level]
        scale = tuple(a / b for a, b in zip(target_scale, current_scale))
        ref_patch_size = filter_params["ref_patch_size"]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))

        filter_params = filter_params.copy()
        filter_params["a_t"] = filter_params["a_t"] * scaled_ref_patch_area
        filter_params["a_h"] = filter_params["a_h"] * scaled_ref_patch_area

        # Find and filter contours
        contours, hierarchy = cv2.findContours(
            img_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )  # Find contours
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params:
            # Necessary for filtering out artifacts
            foreground_contours, hole_contours = _filter_contours(
                contours, hierarchy, filter_params
            )

        # scale detected contours to level 0
        self.contours_tissue = self.scaleContourDim(foreground_contours, target_scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, target_scale)

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype="int32") for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [
            [np.array(hole * scale, dtype="int32") for hole in holes]
            for holes in contours
        ]
