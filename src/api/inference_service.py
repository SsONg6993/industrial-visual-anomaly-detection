from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image

from torchvision import models
from torchvision.transforms.functional import (
    pil_to_tensor,
    normalize,
)


class InferenceService:

    TARGET_SIZE = 256
    MIN_VALID_RATIO = 0.50

    def __init__(
        self,
        project_root: Path
    ):

        self.project_root = Path(
            project_root
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.threshold = None
        self.model = None
        self.memory_bank = None

        self.is_ready = False

        self._load_resources()


    # ========================================================
    # Public properties
    # ========================================================

    @property
    def device_name(self) -> str:

        return str(
            self.device
        )


    @property
    def coreset_size(self) -> int:

        if self.memory_bank is None:
            return 0

        return int(
            self.memory_bank.shape[0]
        )


    # ========================================================
    # Load resources
    # ========================================================

    def _load_threshold(
        self
    ) -> float:

        results_path = (
            self.project_root
            / "reports"
            / "clean_evaluation"
            / "clean_v2c_final_results.json"
        )

        if not results_path.exists():

            raise FileNotFoundError(
                f"Clean evaluation results not found:\n"
                f"{results_path}"
            )

        data = json.loads(
            results_path.read_text(
                encoding="utf-8"
            )
        )

        threshold = float(
            data[
                "threshold_selection"
            ][
                "selected_threshold"
            ]
        )

        return threshold


    def _load_resources(
        self
    ) -> None:

        # -----------------------------------------
        # Threshold
        # -----------------------------------------

        self.threshold = (
            self._load_threshold()
        )


        # -----------------------------------------
        # ResNet18
        # -----------------------------------------

        weights = (
            models.ResNet18_Weights.DEFAULT
        )

        self.model = models.resnet18(
            weights=weights
        ).to(
            self.device
        )

        self.model.eval()


        # -----------------------------------------
        # Clean coreset
        # -----------------------------------------

        coreset_path = (
            self.project_root
            / "models"
            / "clean_evaluation"
            / "acc_clean_full32_greedy10_coreset.pt"
        )

        if not coreset_path.exists():

            raise FileNotFoundError(
                f"Coreset not found:\n"
                f"{coreset_path}"
            )

        checkpoint = torch.load(
            coreset_path,
            map_location="cpu",
            weights_only=True
        )

        self.memory_bank = (
            checkpoint[
                "memory_bank"
            ]
            .float()
            .to(
                self.device
            )
        )

        self.memory_bank = F.normalize(
            self.memory_bank,
            p=2,
            dim=1
        )

        self.is_ready = True


    # ========================================================
    # Prediction label
    # ========================================================

    def _label_from_score(
        self,
        score: float
    ) -> str:

        return (
            "defect"
            if score >= self.threshold
            else "good"
        )


    # ========================================================
    # Letterbox preprocessing
    # ========================================================

    def _letterbox_image(
        self,
        image: Image.Image
    ):

        image = image.convert(
            "RGB"
        )

        original_width, original_height = (
            image.size
        )

        scale = min(
            self.TARGET_SIZE
            / original_width,

            self.TARGET_SIZE
            / original_height
        )

        new_width = max(
            1,
            int(
                round(
                    original_width
                    *
                    scale
                )
            )
        )

        new_height = max(
            1,
            int(
                round(
                    original_height
                    *
                    scale
                )
            )
        )

        resized = image.resize(
            (
                new_width,
                new_height
            ),
            resample=Image.Resampling.BILINEAR
        )


        # -----------------------------------------
        # ImageNet mean as padding color
        # -----------------------------------------

        mean_rgb = (
            0.485,
            0.456,
            0.406
        )

        padding_color = tuple(
            int(
                round(
                    value * 255
                )
            )
            for value in mean_rgb
        )


        canvas = Image.new(
            "RGB",
            (
                self.TARGET_SIZE,
                self.TARGET_SIZE
            ),
            color=padding_color
        )


        left = (
            self.TARGET_SIZE
            -
            new_width
        ) // 2

        top = (
            self.TARGET_SIZE
            -
            new_height
        ) // 2


        canvas.paste(
            resized,
            (
                left,
                top
            )
        )


        # -----------------------------------------
        # Valid-region mask
        # -----------------------------------------

        valid_region = torch.zeros(
            (
                1,
                self.TARGET_SIZE,
                self.TARGET_SIZE
            ),
            dtype=torch.float32
        )

        valid_region[
            :,
            top:top + new_height,
            left:left + new_width
        ] = 1.0


        # -----------------------------------------
        # PIL -> Tensor
        # -----------------------------------------

        tensor = (
            pil_to_tensor(
                canvas
            )
            .float()
            /
            255.0
        )


        # -----------------------------------------
        # ImageNet normalization
        # -----------------------------------------

        tensor = normalize(
            tensor,
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )


        return (
            tensor,
            valid_region
        )


    # ========================================================
    # ResNet18 Full32 features
    # ========================================================

    def _extract_highres_features(
        self,
        tensor
    ):

        x = self.model.conv1(
            tensor
        )

        x = self.model.bn1(
            x
        )

        x = self.model.relu(
            x
        )

        x = self.model.maxpool(
            x
        )


        x = self.model.layer1(
            x
        )


        # 256 input
        # layer2 -> ~32x32x128
        layer2 = self.model.layer2(
            x
        )


        # layer3 -> ~16x16x256
        layer3 = self.model.layer3(
            layer2
        )


        # Upsample layer3 to 32x32
        layer3_up = F.interpolate(
            layer3,
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


        # 128 + 256 = 384 dimensions
        combined = torch.cat(
            [
                layer2,
                layer3_up
            ],
            dim=1
        )


        return combined


    # ========================================================
    # Feature map -> local patch features
    # ========================================================

    def _feature_map_to_patches(
        self,
        feature_map
    ):

        _, channels, h, w = (
            feature_map.shape
        )

        patches = (
            feature_map
            .permute(
                0,
                2,
                3,
                1
            )
            .reshape(
                h * w,
                channels
            )
        )


        patches = F.normalize(
            patches,
            p=2,
            dim=1
        )


        return (
            patches,
            h,
            w
        )


    # ========================================================
    # Valid patch filtering
    # ========================================================

    def _valid_region_to_patch_mask(
        self,
        valid_region,
        h,
        w
    ):

        valid_region = (
            valid_region
            .unsqueeze(0)
            .to(
                self.device
            )
        )


        valid_ratio = (
            F.adaptive_avg_pool2d(
                valid_region,
                output_size=(
                    h,
                    w
                )
            )
            .reshape(-1)
        )


        valid_patch_mask = (
            valid_ratio
            >=
            self.MIN_VALID_RATIO
        )


        return valid_patch_mask


    # ========================================================
    # Exact nearest-neighbor anomaly score
    # ========================================================

    def _calculate_anomaly_score(
        self,
        valid_patches
    ) -> float:

        similarity_matrix = (
            valid_patches
            @
            self.memory_bank.T
        )


        nearest_similarity = (
            similarity_matrix
            .max(
                dim=1
            )
            .values
        )


        distance_squared = (
            2.0
            -
            2.0
            *
            nearest_similarity
        )


        distance_squared = torch.clamp(
            distance_squared,
            min=0.0
        )


        patch_distances = torch.sqrt(
            distance_squared
        )


        image_score = (
            patch_distances
            .max()
            .item()
        )


        return float(
            image_score
        )


    # ========================================================
    # Public predict()
    # ========================================================

    def predict(
        self,
        image: Image.Image
    ) -> dict:

        if not self.is_ready:

            raise RuntimeError(
                "Inference service is not ready."
            )


        start_time = (
            time.perf_counter()
        )


        # -----------------------------------------
        # Preprocessing
        # -----------------------------------------

        (
            tensor,
            valid_region

        ) = self._letterbox_image(
            image
        )


        tensor = (
            tensor
            .unsqueeze(0)
            .to(
                self.device
            )
        )


        # -----------------------------------------
        # Model inference
        # -----------------------------------------

        with torch.no_grad():

            feature_map = (
                self._extract_highres_features(
                    tensor
                )
            )


            patches, h, w = (
                self._feature_map_to_patches(
                    feature_map
                )
            )


            valid_patch_mask = (
                self._valid_region_to_patch_mask(
                    valid_region,
                    h,
                    w
                )
            )


            valid_patches = (
                patches[
                    valid_patch_mask
                ]
            )


            if valid_patches.shape[0] == 0:

                raise RuntimeError(
                    "No valid image patches found."
                )


            anomaly_score = (
                self._calculate_anomaly_score(
                    valid_patches
                )
            )


        # -----------------------------------------
        # GPU synchronization
        # -----------------------------------------

        if self.device.type == "cuda":

            torch.cuda.synchronize()


        latency_ms = (
            time.perf_counter()
            -
            start_time
        ) * 1000


        prediction = (
            self._label_from_score(
                anomaly_score
            )
        )


        return {
            "prediction":
                prediction,

            "anomaly_score":
                anomaly_score,

            "threshold":
                float(
                    self.threshold
                ),

            "latency_ms":
                float(
                    latency_ms
                ),
        }