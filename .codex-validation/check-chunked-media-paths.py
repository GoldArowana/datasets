"""Independent offline controls for Arrow media casts using in-memory data."""
import json

import pyarrow as pa
from datasets import Audio, Dataset, Features, Pdf, Video
from datasets.table import table_cast


def cast_media_rows(data, feature):
    table = pa.table({"asset": data})
    assert table.column("asset").num_chunks == 2
    expected_schema = Features({"asset": feature}).arrow_schema
    converted = table_cast(table, expected_schema)
    column = converted.column("asset")
    assert column.type == feature.pa_type
    assert column.null_count == 1
    assert converted.schema.equals(expected_schema, check_metadata=True)
    return Dataset(converted).to_list()


def main():
    path_results = []
    byte_controls = []
    for constructor, extension in ((Audio, "wav"), (Video, "mp4"), (Pdf, "pdf")):
        feature = constructor(decode=False)
        path = f"媒体/é/longer_than_a_stringview_inline_value.{extension}"
        expected = [{"asset": {"path": path, "bytes": None}}, {"asset": None}, {"asset": {"path": "", "bytes": None}}]
        for data_type in (pa.string(), pa.large_string(), pa.string_view()):
            data = pa.chunked_array([[path, None], [""]], type=data_type)
            assert cast_media_rows(data, feature) == expected, (type(feature).__name__, data_type)
            path_results.append((type(feature).__name__, str(data_type)))
        payload = b"in-memory media bytes"
        data = pa.chunked_array([[payload, None], [b""]], type=pa.binary())
        expected_bytes = [
            {"asset": {"path": None, "bytes": payload}},
            {"asset": None},
            {"asset": {"path": None, "bytes": b""}},
        ]
        assert cast_media_rows(data, feature) == expected_bytes, type(feature).__name__
        byte_controls.append(type(feature).__name__)
    print(json.dumps({"pyarrow": pa.__version__, "chunked_path_cases": path_results, "binary_controls": byte_controls}))


if __name__ == "__main__":
    main()
