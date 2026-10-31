"""Colour-region and depth maps for MOSS-Image control adapter training (moss_image_train_network.py --control_adapter),
made as the published adapter's were: Mask2Former Swin-L (COCO panoptic) paints every segment one random flat colour
(the colours are only region IDs; pixels no segment covers stay black, so a subject outside the model's classes keeps
its own black silhouette), Depth Anything V2 gives the relative depth, normalised per image with near = white.
--segmentation_model facebook/mask2former-swin-large-ade-panoptic (ADE20K, 150 classes with "plaything, toy") names
other subjects.

For every image of --image_dir it writes ``<output_dir>/<image name>_0.png`` (colour) and ``<image name>_1.png`` (depth)
at the image's size as stored (EXIF rotation is ignored, as the trainers ignore it), the dataset's control_directory
layout; --dataset_config does that for every dataset of a training config that names a control_directory. Depth Anything
V2 Large is the depth model the published adapter was trained with; --depth_model depth-anything/Depth-Anything-V2-Small-hf
is a smaller, faster one.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from musubi_tuner.dataset.media_utils import glob_images

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SEGMENTATION_MODEL = "facebook/mask2former-swin-large-coco-panoptic"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
COLOUR_SUFFIX, DEPTH_SUFFIX = "_0.png", "_1.png"


def map_paths(output_dir: str, image_path: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(output_dir, stem + COLOUR_SUFFIX), os.path.join(output_dir, stem + DEPTH_SUFFIX)


def region_colours(count: int, seed: int, min_distance: float = 48.0) -> np.ndarray:
    """``count`` random RGB colours, none near black and none close to another (while there is room)."""
    rng = np.random.default_rng(seed)
    colours: list[np.ndarray] = []
    for _ in range(count):
        candidate = rng.integers(40, 256, size=3)
        for _attempt in range(64):
            if all(np.linalg.norm(candidate - other) >= min_distance for other in colours):
                break
            candidate = rng.integers(40, 256, size=3)
        colours.append(candidate)
    return np.asarray(colours, dtype=np.uint8).reshape(-1, 3)


def colour_regions(segmentation: np.ndarray, segment_ids: list[int], seed: int) -> Image.Image:
    """An (H, W) segment-id map as one flat random colour per segment; ids not listed (void) stay black."""
    out = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for segment_id, colour in zip(segment_ids, region_colours(len(segment_ids), seed)):
        out[segmentation == segment_id] = colour
    return Image.fromarray(out, "RGB")


def panoptic_regions(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    size: tuple[int, int],
    threshold: float = 0.8,
    mask_threshold: float = 0.5,
    overlap: float = 0.8,
) -> tuple[torch.Tensor, int]:
    """Mask2Former's panoptic regions of one image: (H, W) segment ids, 0 for void, and the number of segments.

    As transformers' post-processing, the queries whose class (not "no object") scores above ``threshold`` compete for
    each pixel by score x mask probability, and a segment that keeps less than ``overlap`` of its own mask is dropped.
    Unlike it, a pixel joins its winner only where the winner's own mask probability reaches ``mask_threshold``: there
    every pixel goes to some kept segment, so one confident segment covered a whole close-up (the corgi on orange paper,
    "dog") and a subject outside the classes went to the stuff around it (the robot toy, "couch"). Void stays black."""
    probs = class_logits.float().softmax(-1)
    scores, labels = probs.max(-1)
    keep = (labels != class_logits.shape[-1] - 1) & (scores > threshold)
    segmentation = torch.zeros(size, dtype=torch.int32, device=mask_logits.device)
    if not bool(keep.any()):
        return segmentation, 0
    masks = F.interpolate(mask_logits[keep].float()[None], size=size, mode="bilinear", align_corners=False)[0].sigmoid()
    owner = (scores[keep][:, None, None] * masks).argmax(0)
    own = masks.gather(0, owner[None])[0] >= mask_threshold
    count = 0
    for query in range(masks.shape[0]):
        region = (owner == query) & own
        area, full = int(region.sum()), int((masks[query] >= mask_threshold).sum())
        if area == 0 or full == 0 or area / full < overlap:
            continue
        count += 1
        segmentation[region] = count
    return segmentation, count


def normalised_depth(predicted: torch.Tensor, width: int, height: int) -> Image.Image:
    """Depth Anything's relative inverse depth (larger = nearer) at the image's size, min-max normalised: near = white."""
    depth = F.interpolate(predicted.float().reshape(1, 1, *predicted.shape[-2:]), size=(height, width), mode="bicubic", align_corners=False)
    depth = depth[0, 0]
    low, high = float(depth.min()), float(depth.max())
    scaled = (depth - low) / (high - low) if high > low else torch.zeros_like(depth)
    return Image.fromarray((scaled.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy(), "L")


def image_seed(path: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{os.path.basename(path)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


class MapMaker:
    def __init__(
        self,
        device: torch.device,
        segmentation_model: str = SEGMENTATION_MODEL,
        depth_model: str = DEPTH_MODEL,
        max_side: int = 1024,
        models_dir: Optional[str] = None,
    ):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, Mask2FormerForUniversalSegmentation

        self.device = device
        self.max_side = int(max_side)
        cache = {"cache_dir": models_dir} if models_dir else {}  # the models' download folder (default: the Hugging Face cache)
        logger.info(f"Loading {segmentation_model}")
        self.segmentation_processor = AutoImageProcessor.from_pretrained(segmentation_model, **cache)
        self.segmentation = Mask2FormerForUniversalSegmentation.from_pretrained(segmentation_model, **cache).to(device).eval()
        logger.info(f"Loading {depth_model}")
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_model, **cache)
        self.depth = AutoModelForDepthEstimation.from_pretrained(depth_model, **cache).to(device).eval()

    def _working_copy(self, image: Image.Image) -> Image.Image:
        """Segmentation runs at most ``max_side`` on the long side (its masks are upsampled per query); the colour map is
        scaled back with nearest neighbour."""
        scale = self.max_side / max(image.size)
        if scale >= 1:
            return image
        return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)

    @torch.no_grad()
    def colour_map(self, image: Image.Image, seed: int) -> Image.Image:
        work = self._working_copy(image)
        inputs = self.segmentation_processor(images=work, return_tensors="pt").to(self.device)
        outputs = self.segmentation(**inputs)
        segmentation, count = panoptic_regions(outputs.class_queries_logits[0], outputs.masks_queries_logits[0], (work.height, work.width))
        regions = colour_regions(segmentation.cpu().numpy(), list(range(1, count + 1)), seed)
        return regions if regions.size == image.size else regions.resize(image.size, Image.NEAREST)

    @torch.no_grad()
    def depth_map(self, image: Image.Image) -> Image.Image:
        inputs = self.depth_processor(images=image, return_tensors="pt").to(self.device)
        predicted = self.depth(**inputs).predicted_depth
        return normalised_depth(predicted, image.width, image.height)


def dataset_folders(dataset_config: str) -> list[tuple[str, str]]:
    """(image_directory, control_directory) of every dataset of a training config that names both."""
    from musubi_tuner.dataset.config_utils import load_user_config

    config = load_user_config(dataset_config)
    general = config.get("general", {})
    pairs = []
    for dataset in config.get("datasets", []):
        image_dir = dataset.get("image_directory", general.get("image_directory"))
        control_dir = dataset.get("control_directory", general.get("control_directory"))
        if image_dir and control_dir:
            pairs.append((str(image_dir), str(control_dir)))
    return pairs


def make_maps(
    folders: list[tuple[str, str]],
    device: Optional[str] = None,
    segmentation_model: str = SEGMENTATION_MODEL,
    depth_model: str = DEPTH_MODEL,
    max_side: int = 1024,
    skip_existing: bool = False,
    seed: int = 0,
    models_dir: Optional[str] = None,
) -> int:
    """Write both maps for every image of each (image folder, map folder) pair; returns the number of images processed.
    The models load only when an image needs maps."""
    todo = []
    for image_dir, output_dir in folders:
        images = sorted(glob_images(image_dir))
        if not images:
            raise ValueError(f"no images in {image_dir}")
        if os.path.normcase(os.path.abspath(output_dir)) == os.path.normcase(os.path.abspath(image_dir)):
            raise ValueError(f"the map folder must not be the image folder {image_dir} (the maps would train as images): use a control_directory")
        os.makedirs(output_dir, exist_ok=True)
        pending = [path for path in images if not (skip_existing and all(os.path.isfile(p) for p in map_paths(output_dir, path)))]
        logger.info(f"{image_dir}: {len(images)} images, {len(images) - len(pending)} with both maps already")
        todo.extend((path, output_dir) for path in pending)
    if not todo:
        return 0
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    maker = MapMaker(device, segmentation_model, depth_model, max_side, models_dir)
    for index, (path, output_dir) in enumerate(todo, 1):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        colour_path, depth_path = map_paths(output_dir, path)
        maker.colour_map(image, image_seed(path, seed)).save(colour_path)
        maker.depth_map(image).save(depth_path)
        logger.info(f"[{index}/{len(todo)}] {os.path.basename(path)} -> {os.path.basename(colour_path)}, {os.path.basename(depth_path)}")
    return len(todo)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image_dir", default=None, help="the dataset's image folder (with --output_dir)")
    parser.add_argument("--output_dir", default=None, help="where the maps go: the dataset's control_directory")
    parser.add_argument("--dataset_config", default=None, help="a training dataset TOML: maps for every dataset with a control_directory")
    parser.add_argument("--models_dir", default=None, help="download folder of the two models (default: the Hugging Face cache)")
    parser.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda when available)")
    parser.add_argument("--segmentation_model", default=SEGMENTATION_MODEL, help="Hugging Face Mask2Former panoptic model")
    parser.add_argument("--depth_model", default=DEPTH_MODEL, help="Hugging Face depth model (default: Depth Anything V2 Large, the published adapter's)")
    parser.add_argument("--max_side", type=int, default=1024, help="long side the segmentation runs at (the map keeps the image's size)")
    parser.add_argument("--skip_existing", action="store_true", help="leave images that already have both maps")
    parser.add_argument("--seed", type=int, default=0, help="seed of the region colours (per image, from its name)")
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    folders = []
    if args.image_dir or args.output_dir:
        if not (args.image_dir and args.output_dir):
            parser.error("--image_dir and --output_dir go together")
        folders.append((args.image_dir, args.output_dir))
    if args.dataset_config:
        found = dataset_folders(args.dataset_config)
        if not found:
            parser.error(f"no dataset in {args.dataset_config} names a control_directory")
        folders.extend(found)
    if not folders:
        parser.error("give --image_dir and --output_dir, or --dataset_config")
    count = make_maps(
        folders,
        device=args.device,
        segmentation_model=args.segmentation_model,
        depth_model=args.depth_model,
        max_side=args.max_side,
        skip_existing=args.skip_existing,
        seed=args.seed,
        models_dir=args.models_dir,
    )
    logger.info(f"done: maps for {count} images")


if __name__ == "__main__":
    main()
