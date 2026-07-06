"""Bounding-box / annotation validity checks used by the Part 3 quality report."""

from __future__ import annotations

from dataclasses import dataclass, field

from alpr_dataset.annotations.schema import ImageAnnotation


@dataclass
class AnnotationIssues:
    image_path: str
    boxes_outside_image: int = 0
    zero_area_boxes: int = 0
    overlapping_box_pairs: int = 0
    unknown_class_ids: list[int] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.boxes_outside_image
            or self.zero_area_boxes
            or self.overlapping_box_pairs
            or self.unknown_class_ids
        )


def validate_image_annotation(
    ann: ImageAnnotation,
    valid_class_ids: set[int] | None = None,
    overlap_iou_threshold: float = 0.7,
) -> AnnotationIssues:
    """Run all Part-3 structural checks against a single image's annotations."""
    issues = AnnotationIssues(image_path=str(ann.image_path))

    for box in ann.boxes:
        if (
            box.x_min < 0
            or box.y_min < 0
            or box.x_max > ann.image_width
            or box.y_max > ann.image_height
            or box.x_min >= box.x_max
            or box.y_min >= box.y_max
        ):
            issues.boxes_outside_image += 1

        if box.area <= 0:
            issues.zero_area_boxes += 1

        if valid_class_ids is not None and box.class_id not in valid_class_ids:
            issues.unknown_class_ids.append(box.class_id)

    for i in range(len(ann.boxes)):
        for j in range(i + 1, len(ann.boxes)):
            if ann.boxes[i].iou(ann.boxes[j]) >= overlap_iou_threshold:
                issues.overlapping_box_pairs += 1

    return issues


def is_empty_image_annotation(ann: ImageAnnotation) -> bool:
    """True if the image has an annotation file/entry but zero valid boxes."""
    return ann.annotation_path is not None and len(ann.boxes) == 0
