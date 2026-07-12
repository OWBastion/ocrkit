MODEL_OBJECT_PREFIX = "models/pp-ocrv6-small"
USER_OBJECT_PREFIX = "uploads/"


def model_version_prefix(version: str) -> str:
    return f"{MODEL_OBJECT_PREFIX}/{version}"
