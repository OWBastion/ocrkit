from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from training.vision import VisionOcr


def test_vision_requires_optional_macos_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_vision(name: str, *args: object, **kwargs: object) -> object:
        if name == "Vision":
            raise ImportError("Vision unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "Quartz", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSData=object))
    monkeypatch.delitem(sys.modules, "Vision", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing_vision)

    with pytest.raises(RuntimeError, match="Apple Vision requires macOS"):
        VisionOcr()


@pytest.mark.parametrize(
    "bounds",
    [(0.1, 0.2, 0.3, 0.4), ((0.1, 0.2), (0.3, 0.4))],
)
def test_vision_recognize_converts_bottom_left_coordinates(
    monkeypatch: pytest.MonkeyPatch, bounds: object
) -> None:
    configured: dict[str, object] = {}

    class Candidate:
        def string(self) -> str:
            return "  挑战 完成  " 

        def confidence(self) -> float:
            return 0.991

    class Observation:
        def topCandidates_(self, limit: int) -> list[Candidate]:
            assert limit == 1
            return [Candidate()]

        def boundingBox(self) -> object:
            return bounds

    class Request:
        @classmethod
        def alloc(cls) -> "Request":
            return cls()

        def initWithCompletionHandler_(self, completion_handler: object) -> "Request":
            assert callable(completion_handler)
            configured["completion_handler"] = completion_handler
            return self

        def setRecognitionLevel_(self, value: object) -> None:
            configured["level"] = value

        def setRecognitionLanguages_(self, value: list[str]) -> None:
            configured["languages"] = value

        def setAutomaticallyDetectsLanguage_(self, value: bool) -> None:
            configured["detect_language"] = value

        def setUsesLanguageCorrection_(self, value: bool) -> None:
            configured["language_correction"] = value

        def results(self) -> list[Observation]:
            return [Observation()]

    class Handler:
        @classmethod
        def alloc(cls) -> "Handler":
            return cls()

        def initWithCIImage_options_(self, image: object, options: object) -> "Handler":
            assert image == "ci-image"
            assert options is None
            return self

        def performRequests_error_(self, requests: list[Request], error: object) -> tuple[bool, None]:
            assert len(requests) == 1
            assert error is None
            return True, None

    fake_vision = SimpleNamespace(
        VNImageRequestHandler=Handler,
        VNRecognizeTextRequest=Request,
        VNRequestTextRecognitionLevelAccurate="accurate",
    )
    def data_with_bytes(data: bytes, length: int) -> tuple[bytes, int]:
        assert data.startswith(b"\x89PNG")
        assert length == len(data)
        return data, length

    fake_foundation = SimpleNamespace(
        NSData=SimpleNamespace(dataWithBytes_length_=data_with_bytes),
    )
    fake_quartz = SimpleNamespace(
        CIImage=SimpleNamespace(
            imageWithData_=lambda data: "ci-image",
        )
    )
    monkeypatch.setitem(sys.modules, "Vision", fake_vision)
    monkeypatch.setitem(sys.modules, "Foundation", fake_foundation)
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    lines = VisionOcr().recognize(np.zeros((100, 200, 3), dtype=np.uint8))

    assert configured["level"] == "accurate"
    assert configured["languages"] == ["zh-Hans", "en-US"]
    assert configured["detect_language"] is False
    assert configured["language_correction"] is False
    assert callable(configured["completion_handler"])
    assert len(lines) == 1
    assert lines[0].text == "挑战 完成"
    assert lines[0].confidence == 0.991
    assert lines[0].box.tolist() == [[20.0, 40.0], [80.0, 40.0], [80.0, 80.0], [20.0, 80.0]]


def test_vision_recognize_reports_failed_request_without_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Request:
        @classmethod
        def alloc(cls) -> "Request":
            return cls()

        def initWithCompletionHandler_(self, _completion_handler: object) -> "Request":
            return self

        def setRecognitionLevel_(self, _value: object) -> None:
            pass

        def setRecognitionLanguages_(self, _value: list[str]) -> None:
            pass

        def setAutomaticallyDetectsLanguage_(self, _value: bool) -> None:
            pass

        def setUsesLanguageCorrection_(self, _value: bool) -> None:
            pass

    class Handler:
        @classmethod
        def alloc(cls) -> "Handler":
            return cls()

        def initWithCIImage_options_(self, _image: object, _options: object) -> "Handler":
            return self

        def performRequests_error_(self, _requests: list[Request], _error: object) -> tuple[bool, None]:
            return False, None

    fake_vision = SimpleNamespace(
        VNImageRequestHandler=Handler,
        VNRecognizeTextRequest=Request,
        VNRequestTextRecognitionLevelAccurate="accurate",
    )
    monkeypatch.setitem(sys.modules, "Vision", fake_vision)
    monkeypatch.setitem(
        sys.modules,
        "Foundation",
        SimpleNamespace(NSData=SimpleNamespace(dataWithBytes_length_=lambda data, length: data)),
    )
    monkeypatch.setitem(
        sys.modules,
        "Quartz",
        SimpleNamespace(CIImage=SimpleNamespace(imageWithData_=lambda _data: "ci-image")),
    )

    with pytest.raises(RuntimeError, match="no error details"):
        VisionOcr().recognize(np.zeros((20, 20, 3), dtype=np.uint8))
