"""
Convert Roboflow COCO JSON export to YOLO label format.

Usage:
    python -m holdgrasp.scripts.prepare_yolo_data \
        --train-ann data/climbing/annotations/train.json \
        --val-ann   data/climbing/annotations/val.json \
        --image-dir data/climbing/raw \
        --out-dir   data/climbing/train_yolo
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import click
from holdgrasp.models.base import HOLD_CLASSES

_CLASS_TO_IDX = {c: i for i, c in enumerate(HOLD_CLASSES)}


@click.command()
@click.option("--train-ann", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--val-ann", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--image-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(path_type=Path))
def main(train_ann: Path, val_ann: Path, image_dir: Path, out_dir: Path):
    for split, ann_path in [("train", train_ann), ("val", val_ann)]:
        _convert_split(split, ann_path, image_dir, out_dir)
    click.echo("\nDone. Run training on PC (GPU):")
    click.echo("  yolo train cfg=experiments/configs/yolo_finetune.yaml")


def _convert_split(split: str, ann_path: Path, image_dir: Path, out_dir: Path) -> None:
    data = json.loads(ann_path.read_text())
    id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    id_to_size = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    img_out = out_dir / split / "images"
    lbl_out = out_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    labels_by_image: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        cat_name = id_to_name[ann["category_id"]]
        if cat_name not in _CLASS_TO_IDX:
            continue
        w, h = id_to_size[img_id]
        x, y, bw, bh = ann["bbox"]
        # YOLO normalized format: class cx cy bw bh
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        line = f"{_CLASS_TO_IDX[cat_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        labels_by_image.setdefault(img_id, []).append(line)

    for img_id, file_name in id_to_file.items():
        src = image_dir / file_name
        if src.exists():
            shutil.copy2(src, img_out / file_name)
        lbl_file = lbl_out / Path(file_name).with_suffix(".txt").name
        lbl_file.write_text("\n".join(labels_by_image.get(img_id, [])))

    click.echo(f"{split}: {len(id_to_file)} images → {img_out}")


if __name__ == "__main__":
    main()
