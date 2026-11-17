# ruff: noqa: D100, D103
# Scenario names document tests; library definitions require docstrings.
from types import SimpleNamespace as NS

import pytest

from ultraviz.cli import run_cli as main


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--no-summary", "--help"])
    assert exc.value.code == 0
    assert "Ultraviz" in capsys.readouterr().out


def test_video_requires_fps():
    with pytest.raises(SystemExit) as exc:
        main(["--no-summary", "--model", "model.pt", "--source", "in.mp4", "--output", "out.mp4"])
    assert exc.value.code == 2


def test_image(monkeypatch, image, tmp_path):
    result = NS(orig_img=image, names={})
    monkeypatch.setattr("ultraviz.cli.predict", lambda *args, **kwargs: iter([result]))
    output = tmp_path / "image.png"
    assert (
        main(
            [
                "--no-summary",
                "--model",
                "model.pt",
                "--source",
                "in.png",
                "--output",
                str(output),
                "--no-labels",
            ]
        )
        == 0
    )
    assert output.is_file()


def test_multiple_frames_do_not_publish(monkeypatch, image, tmp_path):
    result = NS(orig_img=image, names={})
    monkeypatch.setattr("ultraviz.cli.predict", lambda *args, **kwargs: iter([result, result]))
    output = tmp_path / "image.png"
    with pytest.raises(SystemExit):
        main(
            [
                "--no-summary",
                "--model",
                "model.pt",
                "--source",
                "in.mp4",
                "--output",
                str(output),
                "--no-labels",
            ]
        )
    assert not output.exists()


def test_cli_demo_and_hide_conf_share_the_public_options(monkeypatch, image, tmp_path, capsys):
    import numpy as np

    captured = []
    result = NS(
        orig_img=image,
        names={0: "person"},
        boxes=NS(data=np.array([[20.0, 20.0, 70.0, 70.0, 0.9, 0.0]])),
    )

    def predictions(*args, **options):
        assert options["task"] == "segment"
        return iter([result])

    monkeypatch.setattr("ultraviz.cli.predict", predictions)
    monkeypatch.setattr(
        "ultraviz.annotation.Annotator._labels",
        lambda self, frame, labels: captured.extend(text for text, *_ in labels) or frame,
    )
    assert (
        main(
            [
                "--no-summary",
                "--model",
                "unused.pt",
                "--source",
                "image.jpg",
                "--output",
                str(tmp_path / "out.png"),
                "--demo",
                "segment",
                "--hide-conf",
                "--timing",
                "console",
            ]
        )
        == 0
    )
    assert captured == ["person"]
    assert "inference: n/a" in capsys.readouterr().out


def test_cli_summary_options_and_console_timing(monkeypatch, image, arial, tmp_path, capsys):
    from PIL import Image

    monkeypatch.setattr(
        "ultraviz.cli.predict", lambda *a, **k: iter([NS(orig_img=image, names={})])
    )
    output = tmp_path / "summary.png"
    assert (
        main(
            [
                "--model",
                "unused.pt",
                "--source",
                "in.jpg",
                "--output",
                str(output),
                "--no-labels",
                "--summary",
                "--summary-title",
                "Inspection",
                "--summary-width",
                "100",
                "--summary-columns",
                "1",
                "--timing",
            ]
        )
        == 0
    )
    assert Image.open(output).getbbox() is not None
    assert "visualization:" in capsys.readouterr().out


def test_cli_default_summary_is_visible_in_heatmap_without_flag(
    monkeypatch, image, arial, tmp_path
):
    from ultraviz.summary import SummaryPanel

    captured = []
    original = SummaryPanel.draw

    def capture(self, image, summary, style, font):
        captured.append(
            (summary.task, style.summary, style.boxes, style.labels, style.summary_subtitle)
        )
        return original(self, image, summary, style, font)

    monkeypatch.setattr(SummaryPanel, "draw", capture)
    monkeypatch.setattr(
        "ultraviz.cli.predict", lambda *a, **k: iter([NS(orig_img=image, names={})])
    )
    assert (
        main(
            [
                "--model",
                "unused.pt",
                "--source",
                "in.jpg",
                "--output",
                str(tmp_path / "heatmap.png"),
                "--demo",
                "heatmap",
                "--summary-subtitle",
                "Camera 01",
            ]
        )
        == 0
    )
    assert captured == [("heatmap", True, False, False, "Camera 01")]
