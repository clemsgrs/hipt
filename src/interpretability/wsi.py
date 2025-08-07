import math
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import tqdm
import wholeslidedata as wsd
from PIL import Image


# ignore all warnings from wholeslidedata
warnings.filterwarnings("ignore", module="wholeslidedata")

Image.MAX_IMAGE_PIXELS = 933120000


class SegmentationParameters(NamedTuple):
    """
    Parameters for filtering contours.
    """

    downsample: int  # dowsample factor for loading segmentation mask
    sthresh: int  # segmentation threshold (positive integer, using a higher threshold leads to less foreground and more background detection) (not used when use_otsu=True)
    sthresh_up: int  # upper threshold value for scaling the binary mask
    mthresh: int  # median filter size (positive, odd integer)
    close: int  # additional morphological closing to apply following initial thresholding (positive integer)
    use_otsu: bool  # whether to use Otsu's method for thresholding
    tissue_pixel_value: int  # when loading mask from disk, what pixel value corresponds to tissue


class FilterParameters(NamedTuple):
    """
    Parameters for filtering contours.
    """

    ref_tile_size: int  # reference tile size for filtering
    a_t: int  # contour area threshold for filtering
    a_h: int  # hole area threshold for filtering
    max_n_holes: int  # maximum number of holes allowed


class TilingParameters(NamedTuple):
    """
    Parameters for tiling.
    """

    spacing: float  # spacing at which to tile the slide, in microns per pixel
    tolerance: float  # for matching the spacing, deciding how much spacing can deviate from those specified in the slide metadata.
    tile_size: int  # size of the tiles to extract, in pixels
    overlap: float  # overlap between tiles
    min_tissue_percentage: float  # minimum tissue percentage required for a tile
    drop_holes: bool  # whether to drop tiles that fall within holes
    use_padding: bool  # whether to use padding for tiles at the edges


class WholeSlideImage(object):
    """
    A class for handling Whole Slide Images (wsi) and tile extraction.
    Attributes:
        path (Path): Full path to the wsi.
        name (str): Name of the wsi (stem of the path).
        fmt (str): File format of the wsi.
        wsi (wsd.WholeSlideImage): wsi object.
        spacing_at_level_0 (float): Manually set spacing at level 0.
        spacings (list[float]): List of spacings for each level.
        level_dimensions (list[tuple[int, int]]): Dimensions at each level.
        level_downsamples (list[tuple[float, float]]): Downsample factors for each level.
        backend (str): Backend used for opening the wsi (default: "asap").
        mask_path (Path, optional): Path to the segmentation mask.
        mask (wsd.WholeSlideImage, optional): Segmentation mask object.
        seg_level (int): Level for segmentation.
        binary_mask (np.ndarray): Binary segmentation mask as a numpy array.
    """

    def __init__(
        self,
        path: Path,
        mask_path: Path | None = None,
        spacing_at_level_0: float | None = None,
        backend: str = "asap",
        segment: bool = False,
        segment_params: SegmentationParameters | None = None,
    ):
        """
        Initializes a Whole Slide Image object with optional mask and spacing.

        Args:
            path (Path): Path to the wsi.
            mask_path (Path, optional): Path to the tissue mask, if available. Defaults to None.
            spacing_at_level_0 (float, optional): Manually set spacing at level 0, if speficied. Defaults to None.
            backend (str): Backend to use for opening the wsi. Defaults to "asap".
            segment (bool): Whether to segment the slide if tissue mask is not provided. Defaults to False.
            segment_params (NamedTuple, optional): Segmentation parameters. Used for either loading an existing mask or segmenting the slide.
        """

        self.path = path
        self.name = path.stem.replace(" ", "_")
        self.fmt = path.suffix
        self.wsi = wsd.WholeSlideImage(path, backend=backend)

        self._scaled_contours_cache = {}  # add a cache for scaled contours
        self._scaled_holes_cache = {}  # add a cache for scaled holes
        self._level_spacing_cache = {}  # add a cache for level spacings

        self.spacing_at_level_0 = spacing_at_level_0  # manually set spacing at level 0
        self.spacings = self.get_spacings()
        self.level_dimensions = self.wsi.shapes
        self.level_downsamples = self.get_downsamples()
        self.backend = backend

        self.mask_path = mask_path
        if mask_path is not None:
            self.mask = wsd.WholeSlideImage(mask_path, backend=backend)
            self.seg_level = self.load_segmentation(segment_params)
        elif segment:
            self.seg_level = self.segment_tissue(segment_params)

    def get_slide(self, spacing: float):
        return self.wsi.get_slide(spacing=spacing)

    def get_tile(self, x: int, y: int, width: int, height: int, spacing: float):
        """
        Extracts a tile from a whole slide image at the specified coordinates, size, and spacing.

        Args:
            x (int): The x-coordinate of the top-left corner of the tile.
            y (int): The y-coordinate of the top-left corner of the tile.
            width (int): Tile width.
            height (int): Tile height.
            spacing (float): The spacing (resolution) at which the tile should be extracted.

        Returns:
            numpy.ndarray: The extracted tile as a numpy array.
        """
        return self.wsi.get_patch(
            x,
            y,
            width,
            height,
            spacing=spacing,
            center=False,
        )

    def get_downsamples(self):
        """
        Calculate the downsample factors for each level of the image pyramid.

        This method computes the downsample factors for each level in the image
        pyramid relative to the base level (level 0). The downsample factor for
        each level is represented as a tuple of two values, corresponding to the
        downsampling in the width and height dimensions.

        Returns:
            list of tuple: A list of tuples where each tuple contains two float
            values representing the downsample factors (width_factor, height_factor)
            for each level relative to the base level.
        """
        level_downsamples = []
        dim_0 = self.level_dimensions[0]
        for dim in self.level_dimensions:
            level_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(level_downsample)
        return level_downsamples

    def get_spacings(self):
        """
        Retrieve the spacings for the whole slide image.

        If the `spacing` attribute is not set, the method returns the original spacings
        from the wsi. Otherwise, it calculates adjusted spacings based on the provided
        `spacing` value and the original spacings.

        Returns:
            list: A list of spacings, either the original or adjusted based on the
            `spacing` attribute.
        """
        if self.spacing_at_level_0 is None:
            spacings = self.wsi.spacings
        else:
            spacings = [
                self.spacing_at_level_0 * s / self.wsi.spacings[0]
                for s in self.wsi.spacings
            ]
        return spacings

    def get_level_spacing(self, level: int):
        """
        Retrieve the spacing value for a specified level.

        Args:
            level (int): Level for which to retrieve the spacing.

        Returns:
            float: Spacing value corresponding to the specified level.
        """
        if level not in self._level_spacing_cache:
            self._level_spacing_cache[level] = self.spacings[level]
        return self._level_spacing_cache[level]

    def get_best_level_for_spacing(
        self, target_spacing: float, tolerance: float
    ):
        """
        Determines the best level in a multi-resolution image pyramid for a given target spacing.

        Ensures that the spacing of the returned level is either within the specified tolerance of the target
        spacing or smaller than the target spacing to avoid upsampling.

        Args:
            target_spacing (float): Desired spacing.
            tolerance (float, optional): Tolerance for matching the spacing, deciding how much
                spacing can deviate from those specified in the slide metadata.

        Returns:
            level (int): Index of the best matching level in the image pyramid.
        """
        spacing = self.get_level_spacing(0)
        target_downsample = target_spacing / spacing
        level = self.get_best_level_for_downsample_custom(target_downsample)
        level_spacing = self.get_level_spacing(level)

        # check if the level_spacing is within the tolerance of the target_spacing
        is_within_tolerance = False
        if abs(level_spacing - target_spacing) / target_spacing <= tolerance:
            is_within_tolerance = True
            return level, is_within_tolerance

        # otherwise, look for a spacing smaller than or equal to the target_spacing
        else:
            while level > 0 and level_spacing > target_spacing:
                level -= 1
                level_spacing = self.get_level_spacing(level)
                if abs(level_spacing - target_spacing) / target_spacing <= tolerance:
                    is_within_tolerance = True
                    break

        assert (
            level_spacing <= target_spacing
            or abs(level_spacing - target_spacing) / target_spacing <= tolerance
        ), f"Unable to find a spacing less than or equal to the target spacing ({target_spacing}) or within {int(tolerance * 100)}% of the target spacing."
        return level, is_within_tolerance

    # def get_best_level_for_downsample_custom(self, downsample: float | int):
    def get_best_level_for_downsample_custom(self, downsample: int):
        """
        Determines the best level for a given downsample factor based on the available
        level downsample values.

        Args:
            downsample (float): Target downsample factor.

        Returns:
            int: Index of the best matching level for the given downsample factor.
        """
        level = int(np.argmin([abs(x - downsample) for x, _ in self.level_downsamples]))
        return level

    def load_segmentation(
        self,
        segment_params: SegmentationParameters,
    ):
        """
        Load and process a segmentation mask for a whole slide image.

        This method ensures that the segmentation mask and the slide have at least one
        common spacing, determines the best level for the given downsample factor, and
        processes the segmentation mask to create a binary mask.

        Args:
            downsample (int): Downsample factor for finding best level for tissue segmentation.
            sthresh_up (int, optional): Upper threshold value for scaling the binary
                mask. Defaults to 255.
            tissue_pixel_value (int, optional): Pixel value in the segmentation mask that
                represents tissue. Defaults to 1.

        Returns:
            int: Level at which the tissue mask was loaded.
        """
        mask_spacing_at_level_0 = self.mask.spacings[0]
        seg_level = self.get_best_level_for_downsample_custom(segment_params.downsample)
        seg_spacing = self.get_level_spacing(seg_level)

        mask_downsample = seg_spacing / mask_spacing_at_level_0
        mask_level = int(
            np.argmin([abs(x - mask_downsample) for x in self.mask.downsamplings])
        )
        mask_spacing = self.mask.spacings[mask_level]

        scale = seg_spacing / mask_spacing
        while scale < 1 and mask_level > 0:
            mask_level -= 1
            mask_spacing = self.mask.spacings[mask_level]
            scale = seg_spacing / mask_spacing

        mask = self.mask.get_slide(spacing=mask_spacing)
        width, height, _ = mask.shape

        # resize the mask to the size of the slide at seg_spacing
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(round(height / scale, 0)), int(round(width / scale, 0))),
            interpolation=cv2.INTER_NEAREST,
        )

        m = (mask == segment_params.tissue_pixel_value).astype("uint8")
        if np.max(m) <= 1:
            m = m * segment_params.sthresh_up

        self.binary_mask = m
        return seg_level

    def segment_tissue(
        self,
        segment_params: SegmentationParameters,
    ):
        """
        Segment the tissue via HSV -> Median thresholding -> Binary thresholding -> Morphological closing.

        Args:
            downsample (int): Downsample factor for finding best level for tissue segmentation.
            sthresh (int, optional): Lower threshold for binary thresholding. Defaults to 20.
            sthresh_up (int, optional): Upper threshold for binary thresholding. Defaults to 255.
            mthresh (int, optional): Kernel size for median blurring. Defaults to 7.
            close (int, optional): Size of the kernel for morphological closing.
                If 0, no morphological closing is applied. Defaults to 0.
            use_otsu (bool, optional): Whether to use Otsu's method for thresholding. Defaults to False.

        Returns:
            int: Level at which the tissue mask was created.
        """

        seg_level = self.get_best_level_for_downsample_custom(segment_params.downsample)
        seg_spacing = self.get_level_spacing(seg_level)

        img = self.wsi.get_slide(spacing=seg_spacing)
        img = np.array(Image.fromarray(img).convert("RGBA"))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert to HSV space
        img_med = cv2.medianBlur(
            img_hsv[:, :, 1], segment_params.mthresh
        )  # apply median blurring

        # thresholding
        if segment_params.use_otsu:
            _, img_thresh = cv2.threshold(
                img_med,
                0,
                segment_params.sthresh_up,
                cv2.THRESH_OTSU + cv2.THRESH_BINARY,
            )
        else:
            _, img_thresh = cv2.threshold(
                img_med,
                segment_params.sthresh,
                segment_params.sthresh_up,
                cv2.THRESH_BINARY,
            )

        # morphological closing
        if segment_params.close > 0:
            kernel = np.ones((segment_params.close, segment_params.close), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        self.binary_mask = img_thresh
        return seg_level

    def visualize_mask(
        self,
        contours,
        holes,
        downsample: int = 32,
        color: tuple[int] = (0, 255, 0),
        hole_color: tuple[int] = (0, 0, 255),
        line_thickness: int = 250,
        # max_size: int | None = None,
        max_size: int = None,
        number_contours: bool = False,
    ):
        vis_level = self.get_best_level_for_downsample_custom(downsample)
        level_downsample = self.level_downsamples[vis_level]
        scale = [1 / level_downsample[0], 1 / level_downsample[1]]

        s = self.spacings[vis_level]
        img = self.wsi.get_slide(spacing=s)
        if self.backend == "openslide":
            img = np.ascontiguousarray(img)

        offset = tuple(-(np.array((0, 0)) * scale).astype(int))
        line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
        if contours is not None:
            if not number_contours:
                cv2.drawContours(
                    img,
                    self.scaleContourDim(contours, scale),
                    -1,
                    color,
                    line_thickness,
                    lineType=cv2.LINE_8,
                    offset=offset,
                )

            else:
                # add numbering to each contour
                for idx, cont in enumerate(contours):
                    contour = np.array(self.scaleContourDim(cont, scale))
                    M = cv2.moments(contour)
                    cX = int(M["m10"] / (M["m00"] + 1e-9))
                    cY = int(M["m01"] / (M["m00"] + 1e-9))
                    # draw the contour and put text next to center
                    cv2.drawContours(
                        img,
                        [contour],
                        -1,
                        color,
                        line_thickness,
                        lineType=cv2.LINE_8,
                        offset=offset,
                    )
                    cv2.putText(
                        img,
                        "{}".format(idx),
                        (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 0),
                        10,
                    )

            for holes in holes:
                cv2.drawContours(
                    img,
                    self.scaleContourDim(holes, scale),
                    -1,
                    hole_color,
                    line_thickness,
                    lineType=cv2.LINE_8,
                )

        img = Image.fromarray(img)

        w, h = img.size
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    def get_tile_coordinates(
        self,
        tiling_params: TilingParameters,
        filter_params: FilterParameters,
        num_workers: int = 1,
    ):
        """
        Extract tile coordinates based on the specified target spacing, tile size, overlap,
        and additional tiling and filtering parameters.

        Args:
            tiling_params (NamedTuple): Parameters for tiling, including:
                - spacing (float): Desired spacing of the tiles.
                - tolerance (float): Tolerance for matching the spacing, deciding how much
                    spacing can deviate from those specified in the slide metadata.
                - tile_size (int): Desired size of the tiles at the target spacing.
                - overlap (float, optional): Overlap between adjacent tiles. Defaults to 0.0.
                - "drop_holes" (bool): If True, tiles falling within a hole will be excluded. Defaults to False.
                - "min_tissue_percentage" (float): Minimum amount pixels covered with tissue required for a tile. Defaults to 0.25 (25 percent).
                - "use_padding" (bool): Whether to use padding for tiles at the edges. Defaults to True.
            filter_params (NamedTuple): Parameters for filtering contours, including:
                - "ref_tile_size" (int): Reference tile size for filtering. Defaults to 256.
                - "a_t" (int): Contour area threshold for filtering. Defaults to 4.
                - "a_h" (int): Hole area threshold for filtering. Defaults to 2.
                - "max_n_holes" (int): Maximum number of holes allowed. Defaults to 8.
            num_workers (int, optional): Number of workers to use for parallel processing.
                Defaults to 1.

        Returns:
            tuple:
                - tile_coordinates (list[tuple[int, int]]): List of (x, y) coordinates for the extracted tiles.
                - tissue_percentages (list[float]): List of tissue percentages for each tile.
                - tile_level (int): Level of the wsi used for tile extraction.
                - resize_factor (float): The factor by which the tile size was resized.
                - tile_size_lv0 (int): The tile size at level 0 of the wsi pyramid.
        """
        scale = tiling_params.spacing / self.get_level_spacing(0)
        tile_size_lv0 = int(round(tiling_params.tile_size * scale, 0))

        contours, holes = self.detect_contours(
            target_spacing=tiling_params.spacing,
            tolerance=tiling_params.tolerance,
            filter_params=filter_params
        )
        (
            running_x_coords,
            running_y_coords,
            tissue_percentages,
            tile_level,
            resize_factor,
        ) = self.process_contours(
            contours,
            holes,
            spacing=tiling_params.spacing,
            tolerance=tiling_params.tolerance,
            tile_size=tiling_params.tile_size,
            overlap=tiling_params.overlap,
            drop_holes=tiling_params.drop_holes,
            min_tissue_percentage=tiling_params.min_tissue_percentage,
            use_padding=tiling_params.use_padding,
            num_workers=num_workers,
        )
        tile_coordinates = list(zip(running_x_coords, running_y_coords))
        return (
            contours,
            holes,
            tile_coordinates,
            tissue_percentages,
            tile_level,
            resize_factor,
            tile_size_lv0,
        )

    @staticmethod
    def filter_contours(contours, hierarchy, filter_params: FilterParameters):
        """
        Filter contours by area using FilterParameters.
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
            if a > filter_params.a_t:  # Use named tuple instead of dictionary
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
            unfilered_holes = unfilered_holes[
                : filter_params.max_n_holes
            ]  # Use named tuple
            filtered_holes = []

            # filter these holes
            for hole in unfilered_holes:
                if cv2.contourArea(hole) > filter_params.a_h:  # Use named tuple
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        return foreground_contours, hole_contours

    def detect_contours(
        self,
        target_spacing: float,
        tolerance: float,
        filter_params: FilterParameters,
    ):
        """
        Detect and filter contours from a binary mask based on specified parameters.

        This method identifies contours in a binary mask, filters them based on area
        thresholds, and scales the contours to a specified target resolution.

        Args:
            target_spacing (float): Desired spacing at which tiles should be extracted.
            tolerance (float): Tolerance for matching the spacing, deciding how much
                spacing can deviate from those specified in the slide metadata.
            filter_params (NamedTuple): A NamedTuple containing filtering parameters:
                - "a_t" (int): Minimum area threshold for foreground contours.
                - "a_h" (int): Minimum area threshold for holes within contours.
                - "max_n_holes" (int): Maximum number of holes to retain per contour.
                - "ref_tile_size" (int): Reference tile size for computing areas.

        Returns:
            tuple[list[np.ndarray], list[list[np.ndarray]]]:
                - A list of scaled foreground contours.
                - A list of lists containing scaled hole contours for each foreground contour.
        """

        spacing_level, _ = self.get_best_level_for_spacing(target_spacing, tolerance)
        current_scale = self.level_downsamples[spacing_level]
        target_scale = self.level_downsamples[self.seg_level]
        scale = tuple(a / b for a, b in zip(target_scale, current_scale))
        ref_tile_size = filter_params.ref_tile_size
        scaled_ref_tile_area = int(round(ref_tile_size**2 / (scale[0] * scale[1]),0))

        adjusted_filter_params = FilterParameters(
            ref_tile_size=filter_params.ref_tile_size,
            a_t=filter_params.a_t * scaled_ref_tile_area,
            a_h=filter_params.a_h * scaled_ref_tile_area,
            max_n_holes=filter_params.max_n_holes,
        )

        # find and filter contours
        contours, hierarchy = cv2.findContours(
            self.binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        foreground_contours, hole_contours = self.filter_contours(
            contours, hierarchy, adjusted_filter_params
        )

        # scale detected contours to level 0
        contours = self.scaleContourDim(foreground_contours, target_scale)
        holes = self.scaleHolesDim(hole_contours, target_scale)
        return contours, holes

    @staticmethod
    def isInHoles(holes, pt, tile_size):
        """
        Check if a given tile is inside any of the specified polygonal holes.

        Args:
            holes (list): A list of polygonal contours, where each contour is represented
                        as a list of points (e.g., from OpenCV's findContours function).
            pt (tuple): The (x, y) coordinates of the top-left corner of the tile to check.
            tile_size (int or float): The size of the tile, used to calculate the center
                                    of the point being tested.

        Returns:
            int: Returns 1 if the point is inside any of the holes, otherwise returns 0.
        """
        for hole in holes:
            if (
                cv2.pointPolygonTest(
                    hole, (pt[0] + tile_size / 2, pt[1] + tile_size / 2), False
                )
                > 0
            ):
                return 1

        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, drop_holes=True, tile_size=256):
        """
        Determines whether a given tile is within contours (and optionally outside of holes).

        Args:
            cont_check_fn (callable): A function that checks if a tile is within contours.
                It should accept a (x,y) coordinates as input and return a tuple (keep_flag, tissue_pct),
                where `keep_flag` is a boolean indicating if the tile is within contours,
                and `tissue_pct` is the percentage of tissue coverage of the tile.
            pt (tuple): The (x, y) coordinates of the top-left corner of the tile to check.
            holes (list, optional): A list of holes (e.g., regions to exclude) to check against.
                Defaults to None.
            drop_holes (bool, optional): If True, tiles falling within a hole will be excluded.
                Defaults to True.
            tile_size (int, optional): The size of the tile to consider.
                Defaults to 256.

        Returns:
            tuple: A tuple (keep_flag, tissue_pct), where:
                - `keep_flag` is 1 if the tile is within contours and not in holes (if applicable),
                  otherwise 0.
                - `tissue_pct` is the percentage of tissue coverage of the tile.
        """
        keep_flag, tissue_pct = cont_check_fn(pt)
        if keep_flag:
            if holes is not None and drop_holes:
                return not WholeSlideImage.isInHoles(holes, pt, tile_size), tissue_pct
            else:
                return 1, tissue_pct
        return 0, tissue_pct

    @staticmethod
    def scaleContourDim(contours, scale):
        """
        Scales the dimensions of a list of contours by a given factor.

        Args:
            contours (list of numpy.ndarray): A list of contours, where each contour is
                represented as a numpy array of coordinates.
            scale (float): The scaling factor to apply to the contours.

        Returns:
            list of numpy.ndarray: A list of scaled contours, where each contour's
            coordinates are multiplied by the scaling factor and converted to integers.
        """
        return [np.array(cont * scale, dtype="int32") for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        """
        Scales the dimensions of holes within a set of contours by a given factor.

        Args:
            contours (list of list of numpy.ndarray): A list of contours, where each contour
                is represented as a list of holes, and each hole is a numpy array of coordinates.
            scale (float): The scaling factor to apply to the dimensions of the holes.

        Returns:
            list of list of numpy.ndarray: A new list of contours with the dimensions of
            the holes scaled by the specified factor.
        """
        return [
            [np.array(hole * scale, dtype="int32") for hole in holes]
            for holes in contours
        ]

    def process_contours(
        self,
        contours,
        holes,
        spacing: float,
        tolerance: float,
        tile_size: int,
        overlap: float,
        drop_holes: bool,
        min_tissue_percentage: float,
        use_padding: bool,
        num_workers: int = 1,
    ):
        """
        Processes a list of contours and their corresponding holes to generate tile coordinates,
        tissue percentages, and other metadata.

        Args:
            contours (list): List of contours representing tissue blobs in the wsi.
            holes (list): List of tissue holes in each contour.
            spacing (float): Desired spacing for tiling.
            tolerance (float): Tolerance for matching the spacing, deciding how much
                spacing can deviate from those specified in the slide metadata.
            tile_size (int): Desired tile size in pixels.
            overlap (float): Overlap between adjacent tiles.
            drop_holes (bool): Whether to drop tiles that fall within holes.
            min_tissue_percentage (float): Minimum amount pixels covered with tissue required for a tile.
            use_padding (bool): Whether to pad the tiles to ensure full coverage.
            num_workers (int, optional): Number of workers to use for parallel processing. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - running_x_coords (list): The x-coordinates of the extracted tiles.
                - running_y_coords (list): The y-coordinates of the extracted tiles.
                - running_tissue_pct (list): List of tissue percentages for each extracted tile.
                - tile_level (int): Level of the wsi used for tile extraction.
                - resize_factor (float): The factor by which the tile size was resized.
        """
        running_x_coords, running_y_coords = [], []
        running_tissue_pct = []
        tile_level = None
        resize_factor = None

        def process_single_contour(i):
            return self.process_contour(
                contours[i],
                holes[i],
                spacing,
                tolerance,
                tile_size,
                overlap,
                drop_holes,
                min_tissue_percentage,
                use_padding,
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm.tqdm(
                    executor.map(process_single_contour, range(len(contours))),
                    desc="Extracting tissue tiles",
                    unit=" tissue blob",
                    total=len(contours),
                    leave=True,
                    file=sys.stdout,
                )
            )

        for (
            x_coords,
            y_coords,
            tissue_pct,
            cont_tile_level,
            cont_resize_factor,
        ) in results:
            if len(x_coords) > 0:
                if tile_level is not None:
                    assert (
                        tile_level == cont_tile_level
                    ), "Tile level should be the same for all contours"
                tile_level = cont_tile_level
                resize_factor = cont_resize_factor
                running_x_coords.extend(x_coords)
                running_y_coords.extend(y_coords)
                running_tissue_pct.extend(tissue_pct)

        return (
            running_x_coords,
            running_y_coords,
            running_tissue_pct,
            tile_level,
            resize_factor,
        )

    def process_contour(
        self,
        contour,
        contour_holes,
        spacing: float,
        tolerance: float,
        tile_size: int,
        overlap: float,
        drop_holes: bool,
        min_tissue_percentage: float,
        use_padding: bool,
    ):
        """
        Processes a contour to generate tile coordinates and associated metadata.

        Args:
            contour (numpy.ndarray): Contour to process, defined as a set of points.
            contour_holes (list): List of holes within the contour.
            spacing (float): Target spacing for the tiles.
            tolerance (float): Tolerance for matching the spacing, deciding how much
                spacing can deviate from those specified in the slide metadata.
            tile_size (int): Size of the tiles in pixels.
            overlap (float): Overlap between tiles.
            drop_holes (bool): Whether to drop tiles that fall within holes.
            min_tissue_percentage (float): Minimum amount pixels covered with tissue required for a tile.
            use_padding (bool): Whether to pad the image to ensure full coverage.

        Returns:
            tuple: A tuple containing:
                - x_coords (list): List of x-coordinates for each tile.
                - y_coords (list): List of y-coordinates for each tile.
                - filtered_tissue_percentages (list): List of tissue percentages for each tile.
                - tile_level (int): Level of the image used for tile extraction.
                - resize_factor (float): The factor by which the tile size was resized.
        """
        tile_level, is_within_tolerance = self.get_best_level_for_spacing(
            spacing, tolerance
        )
        tile_spacing = self.get_level_spacing(tile_level)
        resize_factor = spacing / tile_spacing
        if is_within_tolerance:
            resize_factor = 1.0

        assert (
            resize_factor >= 1
        ), f"Resize factor should be greater than or equal to 1. Got {resize_factor}"

        tile_size_resized = int(round(tile_size * resize_factor,0))
        step_size = int(tile_size_resized * (1.0 - overlap))

        if contour is not None:
            start_x, start_y, w, h = cv2.boundingRect(contour)
        else:
            start_x, start_y, w, h = (
                0,
                0,
                self.level_dimensions[tile_level][0],
                self.level_dimensions[tile_level][1],
            )

        tile_downsample = (
            int(self.level_downsamples[tile_level][0]),
            int(self.level_downsamples[tile_level][1]),
        )
        ref_tile_size = (
            tile_size_resized * tile_downsample[0],
            tile_size_resized * tile_downsample[1],
        )

        img_w, img_h = self.level_dimensions[0]
        if use_padding:
            stop_y = int(start_y + h)
            stop_x = int(start_x + w)
        else:
            stop_y = min(start_y + h, img_h - ref_tile_size[1] + 1)
            stop_x = min(start_x + w, img_w - ref_tile_size[0] + 1)

        scale = self.level_downsamples[self.seg_level]
        cont = self.scaleContourDim([contour], (1.0 / scale[0], 1.0 / scale[1]))[0]

        tissue_checker = HasEnoughTissue(
            contour=cont,
            contour_holes=contour_holes,
            tissue_mask=self.binary_mask,
            tile_size=ref_tile_size[0],
            scale=scale,
            pct=min_tissue_percentage,
        )

        ref_step_size_x = int(round(step_size * tile_downsample[0], 0))
        ref_step_size_y = int(round(step_size * tile_downsample[1], 0))

        x_range = np.arange(start_x, stop_x, step=ref_step_size_x)
        y_range = np.arange(start_y, stop_y, step=ref_step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
        coord_candidates = np.array(
            [x_coords.flatten(), y_coords.flatten()]
        ).transpose()

        # vectorized processing of coordinates using the tissue_checker
        keep_flags, tissue_pcts = tissue_checker.check_coordinates(coord_candidates)

        if drop_holes:
            keep_flags = [
                flag and not self.isInHoles(contour_holes, coord, ref_tile_size[0])
                for flag, coord in zip(keep_flags, coord_candidates)
            ]

        filtered_coordinates = coord_candidates[np.array(keep_flags) == 1]
        filtered_tissue_percentages = np.array(tissue_pcts)[np.array(keep_flags) == 1]

        ntile = len(filtered_coordinates)

        if ntile > 0:
            x_coords = list(filtered_coordinates[:, 0])
            y_coords = list(filtered_coordinates[:, 1])
            return (
                x_coords,
                y_coords,
                filtered_tissue_percentages,
                tile_level,
                resize_factor,
            )

        else:
            return [], [], [], None, None

    @staticmethod
    def process_coord_candidate(
        coord, contour_holes, tile_size, cont_check_fn, drop_holes
    ):
        """
        Processes a candidate coordinate to determine if it should be kept based on
        its location relative to contours and the percentage of tissue it contains.

        Args:
            coord (tuple): (x, y) coordinate to be processed.
            contour_holes (list): A list of contours and holes to check against.
            tile_size (int): Size of the tile to consider.
            cont_check_fn (callable): A function to check if the coordinate is within
                the contours or holes.
            drop_holes (bool): A flag indicating whether to drop tiles falling in holes during the check.

        Returns:
            tuple: A tuple containing:
                - coord (tuple or None): Input coordinate if it passes the check,
                otherwise None.
                - tissue_pct (float): Percentage of tissue in the tile.
        """
        keep_flag, tissue_pct = WholeSlideImage.isInContours(
            cont_check_fn, coord, contour_holes, drop_holes, tile_size
        )
        if keep_flag:
            return coord, tissue_pct
        else:
            return None, tissue_pct


class HasEnoughTissue(object):
    def __init__(self, contour, contour_holes, tissue_mask, tile_size, scale, pct=0.01):
        self.cont = contour
        self.holes = contour_holes
        self.mask = tissue_mask // 255
        self.tile_size = tile_size
        self.scale = scale
        self.pct = pct

        # Precompute the combined tissue mask
        self.precomputed_mask = self._precompute_tissue_mask()

    def _precompute_tissue_mask(self):
        """
        Precompute a binary mask for the entire region, combining the contour and holes.

        Returns:
            np.ndarray: A binary mask where tissue regions are 1 and non-tissue regions are 0.
        """
        contour_mask = np.zeros_like(self.mask, dtype=np.uint8)

        # Draw white filled contour on black background
        cv2.drawContours(contour_mask, [self.cont], 0, 255, -1)

        # Draw black filled holes on white filled contour
        cv2.drawContours(contour_mask, self.holes, 0, 0, -1)

        # Combine with the tissue mask
        return cv2.bitwise_and(self.mask, contour_mask)

    def __call__(self, pt):
        """
        Check if a single tile at the given point has enough tissue.

        Args:
            pt (tuple): The (x, y) coordinates of the top-left corner of the tile.

        Returns:
            tuple: (keep_flag, tissue_pct), where:
                - keep_flag is 1 if the tile has enough tissue, otherwise 0.
                - tissue_pct is the percentage of tissue in the tile.
        """
        downsampled_tile_size = int(round(self.tile_size * 1 / self.scale[0], 0))
        assert (
            downsampled_tile_size > 0
        ), "downsampled tile_size is equal to zero, aborting; please consider using a smaller seg_params.downsample parameter"
        downsampled_pt = pt * 1 / self.scale[0]
        x_tile, y_tile = map(int, downsampled_pt)

        # Extract the sub-mask for the tile
        sub_mask = self.precomputed_mask[
            y_tile : y_tile + downsampled_tile_size,
            x_tile : x_tile + downsampled_tile_size,
        ]

        tile_area = downsampled_tile_size**2
        tissue_area = np.sum(sub_mask)
        tissue_pct = round(tissue_area / tile_area, 3)

        if tissue_pct >= self.pct:
            return 1, tissue_pct
        else:
            return 0, tissue_pct

    def check_coordinates(self, coords):
        """
        Check multiple tile coordinates for tissue coverage in a vectorized manner.

        Args:
            coords (np.ndarray): An array of shape (N, 2), where each row is (x, y).

        Returns:
            tuple: (keep_flags, tissue_pcts), where:
                - keep_flags is a list of 1s and 0s indicating whether each tile has enough tissue.
                - tissue_pcts is a list of tissue percentages for each tile.
        """
        downsampled_tile_size = int(round(self.tile_size * 1 / self.scale[0], 0))
        assert (
            downsampled_tile_size > 0
        ), "downsampled tile_size is equal to zero, aborting; please consider using a smaller seg_params.downsample parameter"

        # Downsample coordinates
        downsampled_coords = coords * 1 / self.scale[0]
        downsampled_coords = downsampled_coords.astype(int)

        keep_flags = []
        tissue_pcts = []

        for x_tile, y_tile in downsampled_coords:
            # Extract the sub-mask for the tile
            sub_mask = self.precomputed_mask[
                y_tile : y_tile + downsampled_tile_size,
                x_tile : x_tile + downsampled_tile_size,
            ]

            tile_area = downsampled_tile_size**2
            tissue_area = np.sum(sub_mask)
            tissue_pct = round(tissue_area / tile_area, 3)

            if tissue_pct >= self.pct:
                keep_flags.append(1)
                tissue_pcts.append(tissue_pct)
            else:
                keep_flags.append(0)
                tissue_pcts.append(tissue_pct)

        return keep_flags, tissue_pcts