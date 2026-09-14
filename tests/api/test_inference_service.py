import json
from pathlib import Path

from src.api.inference_service import InferenceService


def test_threshold_is_loaded_from_clean_results(tmp_path: Path):
    report_dir = (
        tmp_path
        / "reports"
        / "clean_evaluation"
    )

    report_dir.mkdir(
        parents=True
    )

    results_path = (
        report_dir
        / "clean_v2c_final_results.json"
    )

    results_path.write_text(
        json.dumps(
            {
                "threshold_selection": {
                    "selected_threshold": 0.569758
                }
            }
        ),
        encoding="utf-8"
    )

    service = InferenceService.__new__(
        InferenceService
    )

    service.project_root = tmp_path

    threshold = service._load_threshold()

    assert threshold == 0.569758

def test_prediction_label_uses_loaded_threshold():

    service = InferenceService.__new__(
        InferenceService
    )

    service.threshold = 0.5

    assert service._label_from_score(
        0.49
    ) == "good"

    assert service._label_from_score(
        0.50
    ) == "defect"

    assert service._label_from_score(
        0.80
    ) == "defect"