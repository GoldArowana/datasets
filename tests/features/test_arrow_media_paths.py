import pyarrow as pa
import pytest

from datasets import Audio, Dataset, Features, Pdf, Video


FEATURES = [
    pytest.param(Audio(decode=False), "sound.wav", id="audio"),
    pytest.param(Video(decode=False), "video.mp4", id="video"),
    pytest.param(Pdf(decode=False), "document.pdf", id="pdf"),
]


@pytest.mark.parametrize("feature,path", FEATURES)
@pytest.mark.parametrize(
    "storage_type", [pa.string(), pa.large_string(), pa.string_view()], ids=["string", "large", "view"]
)
def test_arrow_media_paths(feature, path, storage_type):
    features = Features({"media": feature})
    paths = pa.array([path, None, ""], type=storage_type)

    dataset = Dataset.from_dict({"media": paths}, features=features)

    assert dataset.to_list() == [
        {"media": {"path": path, "bytes": None}},
        {"media": None},
        {"media": {"path": "", "bytes": None}},
    ]
