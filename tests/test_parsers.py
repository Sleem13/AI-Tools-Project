"""Unit tests for annotation parsers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpr_dataset.annotations.parsers import (
    detect_annotation_format,
    parse_annotation_file,
    parse_coco_json,
    parse_csv_annotations,
    parse_voc_xml,
    parse_yolo,
)
from alpr_dataset.annotations.schema import ImageAnnotation


class TestDetectAnnotationFormat:
    def test_yolo_txt(self, tmp_path: Path) -> None:
        p = tmp_path / "labels.txt"
        p.write_text("0 0.5 0.5 0.2 0.1\n")
        assert detect_annotation_format(p) == "yolo"

    def test_voc_xml(self, tmp_path: Path) -> None:
        p = tmp_path / "ann.xml"
        p.write_text("<annotation></annotation>")
        assert detect_annotation_format(p) == "voc_xml"

    def test_coco_json(self, tmp_path: Path) -> None:
        p = tmp_path / "ann.json"
        p.write_text("{}")
        assert detect_annotation_format(p) == "coco_json"

    def test_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "ann.csv"
        p.write_text("filename,xmin\n")
        assert detect_annotation_format(p) == "csv"

    def test_unknown(self, tmp_path: Path) -> None:
        p = tmp_path / "ann.xyz"
        p.write_text("")
        assert detect_annotation_format(p) == "unknown"


class TestParseYolo:
    def test_parses_single_box(self, tmp_yolo_annotation: Path, tmp_image: Path) -> None:
        result = parse_yolo(tmp_yolo_annotation, tmp_image, "test")
        assert result is not None
        assert isinstance(result, ImageAnnotation)
        assert len(result.boxes) == 1
        box = result.boxes[0]
        assert box.class_id == 0
        assert box.class_name == "license_plate"
        # Image is 128x64, YOLO: cx=0.5, cy=0.5, w=0.4, h=0.3
        # x_min = (0.5 - 0.4/2) * 128 = 0.3 * 128 = 38.4
        # y_min = (0.5 - 0.3/2) * 64 = 0.35 * 64 = 22.4
        # x_max = (0.5 + 0.4/2) * 128 = 0.7 * 128 = 89.6
        # y_max = (0.5 + 0.3/2) * 64 = 0.65 * 64 = 41.6
        assert box.x_min == pytest.approx(38.4)
        assert box.y_min == pytest.approx(22.4)
        assert box.x_max == pytest.approx(89.6)
        assert box.y_max == pytest.approx(41.6)

    def test_returns_none_for_unreadable_image(self, tmp_yolo_annotation: Path, tmp_path: Path) -> None:
        bad_image = tmp_path / "nonexistent.jpg"
        result = parse_yolo(tmp_yolo_annotation, bad_image, "test")
        assert result is None

    def test_skips_malformed_lines(self, tmp_path: Path, tmp_image: Path) -> None:
        ann = tmp_path / "bad.txt"
        ann.write_text("0 0.5 0.5\n0 invalid 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n", encoding="utf-8")
        result = parse_yolo(ann, tmp_image, "test")
        assert result is not None
        assert len(result.boxes) == 1

    def test_class_map_remapping(self, tmp_path: Path, tmp_image: Path) -> None:
        ann = tmp_path / "remap.txt"
        ann.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        result = parse_yolo(ann, tmp_image, "test", class_map={0: "car"})
        assert result is not None
        assert result.boxes[0].class_name == "car"


class TestParseVocXml:
    def test_parses_single_object(self, tmp_voc_xml: Path, tmp_image: Path) -> None:
        result = parse_voc_xml(tmp_voc_xml, tmp_image, "test")
        assert result is not None
        assert len(result.boxes) == 1
        box = result.boxes[0]
        assert box.x_min == 10.0
        assert box.y_min == 10.0
        assert box.x_max == 90.0
        assert box.y_max == 50.0
        assert box.class_name == "license_plate"

    def test_returns_none_for_missing_image(self, tmp_voc_xml: Path, tmp_path: Path) -> None:
        result = parse_voc_xml(tmp_voc_xml, tmp_path / "missing.jpg", "test")
        assert result is None

    def test_empty_annotation(self, tmp_path: Path, tmp_image: Path) -> None:
        xml = "<annotation><filename>test_image.jpg</filename><size><width>128</width><height>64</height></size></annotation>"
        p = tmp_path / "empty.xml"
        p.write_text(xml, encoding="utf-8")
        result = parse_voc_xml(p, tmp_image, "test")
        assert result is not None
        assert len(result.boxes) == 0

    def test_skips_object_with_invalid_coordinates(self, tmp_path: Path, tmp_image: Path) -> None:
        xml = """<annotation>
        <object><name>bad</name><bndbox><xmin>invalid</xmin><ymin>1</ymin><xmax>2</xmax><ymax>3</ymax></bndbox></object>
        <object><name>good</name><bndbox><xmin>10</xmin><ymin>11</ymin><xmax>20</xmax><ymax>21</ymax></bndbox></object>
        </annotation>"""
        path = tmp_path / "partially_invalid.xml"
        path.write_text(xml, encoding="utf-8")

        result = parse_voc_xml(path, tmp_image, "test")

        assert result is not None
        assert [box.class_name for box in result.boxes] == ["good"]


class TestParseCocoJson:
    def test_parses_multiple_images(self, tmp_coco_json: Path, tmp_path: Path) -> None:
        results = parse_coco_json(tmp_coco_json, tmp_path, "test")
        assert len(results) == 2
        # First image has 2 annotations
        img1 = next(r for r in results if r.image_path.name == "img1.jpg")
        assert len(img1.boxes) == 2
        assert img1.boxes[0].x_min == 100.0
        assert img1.boxes[0].y_min == 100.0

    def test_returns_empty_for_bad_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        results = parse_coco_json(p, tmp_path, "test")
        assert results == []

    def test_class_names_from_categories(self, tmp_coco_json: Path, tmp_path: Path) -> None:
        results = parse_coco_json(tmp_coco_json, tmp_path, "test")
        assert results[0].boxes[0].class_name == "license_plate"

    def test_skips_malformed_annotation(self, tmp_path: Path) -> None:
        path = tmp_path / "partially_invalid.json"
        path.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "image.jpg", "width": 10, "height": 10}],
                    "categories": [{"id": 1, "name": "license_plate"}],
                    "annotations": [
                        {"image_id": 1, "category_id": 1, "bbox": ["invalid", 0, 1, 1]},
                        {"image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]},
                    ],
                }
            ),
            encoding="utf-8",
        )

        results = parse_coco_json(path, tmp_path, "test")

        assert len(results) == 1
        assert len(results[0].boxes) == 1
        assert results[0].boxes[0].x_min == 1.0


class TestParseCsvAnnotations:
    def test_skips_malformed_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "annotations.csv"
        path.write_text(
            "filename,xmin,ymin,xmax,ymax,width,height,class\n"
            "bad.jpg,invalid,2,3,4,10,10,plate\n"
            "good.jpg,1,2,3,4,10,20,plate\n",
            encoding="utf-8",
        )

        results = parse_csv_annotations(path, tmp_path, "test")

        assert len(results) == 1
        assert results[0].image_path.name == "good.jpg"
        assert (results[0].image_width, results[0].image_height) == (10, 20)


class TestParseAnnotationFile:
    def test_dispatches_to_yolo(self, tmp_yolo_annotation: Path, tmp_image: Path) -> None:
        result = parse_annotation_file(tmp_yolo_annotation, tmp_image, "test")
        assert result is not None
        assert result.source_format == "yolo"

    def test_dispatches_to_voc(self, tmp_voc_xml: Path, tmp_image: Path) -> None:
        result = parse_annotation_file(tmp_voc_xml, tmp_image, "test")
        assert result is not None
        assert result.source_format == "voc_xml"

    def test_returns_none_for_unknown_format(self, tmp_path: Path, tmp_image: Path) -> None:
        p = tmp_path / "file.xyz"
        p.write_text("data")
        result = parse_annotation_file(p, tmp_image, "test", annotation_format="csv")
        assert result is None
