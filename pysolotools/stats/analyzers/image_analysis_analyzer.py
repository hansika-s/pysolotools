import os
from typing import Any, List, Tuple

import cv2
import numpy as np
import pywt
from PIL import Image
from scipy import ndimage

from pysolotools.core.models.solo import (
    BoundingBox2DAnnotation,
    BoundingBox2DLabel,
    Frame,
    RGBCameraCapture,
)
from pysolotools.stats.analyzers.base import AnalyzerBase, AnalyzerFactory


@AnalyzerFactory.register(name="psd")
class PowerSpectrumAnalyzer(AnalyzerBase):
    @staticmethod
    def _load_img(img_path: str):
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = img.convert("L")
        return np.array(img)

    @staticmethod
    def _get_psd2d(image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        fourier_image = np.fft.fft2(image)
        N = h * w * 2
        psd2d = (1 / N) * np.abs(fourier_image) ** 2
        psd2d = np.fft.fftshift(psd2d)
        return psd2d

    @staticmethod
    def _get_psd1d(psd_2d: np.ndarray) -> np.ndarray:
        h = psd_2d.shape[0]
        w = psd_2d.shape[1]
        wc = w // 2
        hc = h // 2

        # create an array of integer radial distances from the center
        y, x = np.ogrid[-h // 2 : h // 2, -w // 2 : w // 2]
        r = np.hypot(x, y).astype(int)
        idx = np.arange(0, min(wc, hc))
        psd_1d = ndimage.sum(psd_2d, r, index=idx)
        return psd_1d

    def analyze(self, frame: Frame = None, **kwargs: Any) -> object:
        solo_data_path = kwargs.get("solo_data_path")
        file_path = os.path.join(solo_data_path, frame.get_file_path(RGBCameraCapture))
        img = self._load_img(file_path)

        psd_2d = self._get_psd2d(img)
        psd_1d = self._get_psd1d(psd_2d)

        return [psd_1d]

    def merge(self, agg_result: Any, frame_result: List, **kwargs: Any) -> List:
        if not agg_result:
            return frame_result

        agg_result += frame_result
        return agg_result


@AnalyzerFactory.register(name="wavelet")
class WaveletTransformAnalyzer(AnalyzerBase):
    def analyze(self, frame: Frame = None, **kwargs: Any) -> object:
        solo_data_path = kwargs.get("solo_data_path")
        file_path = os.path.join(solo_data_path, frame.get_file_path(RGBCameraCapture))
        im = Image.open(file_path).convert("L")
        _, (cH, cV, cD) = pywt.dwt2(im, "haar", mode="periodization")

        return [[cH], [cV], [cD]]

    def merge(self, agg_result: Any, frame_result: Any, **kwargs: Any) -> List:
        if not agg_result:
            return frame_result
        agg_result[0] += frame_result[0]
        agg_result[1] += frame_result[1]
        agg_result[2] += frame_result[2]

        return agg_result


@AnalyzerFactory.register(name="laplacian")
class LaplacianAnalyzer(AnalyzerBase):
    @staticmethod
    def _laplacian_img(img_path: str) -> np.ndarray:
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = laplacian.astype("float")
        return laplacian

    @staticmethod
    def _get_bbox_var_laplacian(
        laplacian: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        bbox_var = laplacian[y : y + h, x : x + w]

        return np.nanvar(bbox_var)

    @staticmethod
    def _get_bbox_fg_bg_var_laplacian(
        laplacian: np.ndarray, annotations: List[BoundingBox2DLabel]
    ) -> Tuple[List, np.ndarray]:

        bbox_var_lap = []
        img_laplacian = laplacian

        for ann in annotations:
            x, y, w, h = [
                int(ann.origin[0]),
                int(ann.origin[1]),
                int(ann.dimension[0]),
                int(ann.dimension[1]),
            ]
            bbox_area = w * h
            if bbox_area >= 1200:  # ignoring small bbox sizes
                bbox_var = LaplacianAnalyzer._get_bbox_var_laplacian(
                    img_laplacian, int(x), int(y), int(w), int(h)
                )
                print(x, y, h, w)
                img_laplacian[int(y) : int(y + h), int(x) : int(x + w)] = np.nan
                bbox_var_lap.append(bbox_var)

        img_var_laplacian = np.nanvar(img_laplacian)

        return bbox_var_lap, img_var_laplacian

    def analyze(self, frame: Frame = None, **kwargs: Any) -> object:
        solo_data_path = kwargs.get("solo_data_path")
        file_path = os.path.join(solo_data_path, frame.get_file_path(RGBCameraCapture))
        laplacian = self._laplacian_img(file_path)
        for capture in filter(
            lambda k: isinstance(k, RGBCameraCapture),
            frame.captures,
        ):
            bbox_anns = list(
                filter(
                    lambda k: isinstance(k, BoundingBox2DAnnotation),
                    capture.annotations,
                )
            )[0].values
            bbox_var_laps, img_var_lap = self._get_bbox_fg_bg_var_laplacian(
                laplacian, bbox_anns
            )

            return [bbox_var_laps, [img_var_lap]]

    def merge(self, agg_result: List, frame_result: List, **kwargs: Any) -> Any:
        if not agg_result:
            return frame_result

        agg_result[0] += frame_result[0]
        agg_result[1] += frame_result[1]

        return agg_result
