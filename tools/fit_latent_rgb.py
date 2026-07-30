"""Fit affine DINAC latent-to-RGB preview factors through ComfyUI's VAE path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT.parents[1]))

import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

from canter_native.vae import load_dinac  # noqa: E402

DEFAULT_BUCKETS = (
    "portrait=pn_hq2_1024/allinone@832x1216",
    "landscape=pn_hq2_1024/allinone@1216x832",
    "square=pn_hq2_1024/allinone@1024x1024",
    "portrait_near=pn_hq3_1024/allinone@832x1152",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Bucket:
    name: str
    directory: Path
    width: int
    height: int


def parse_bucket(specification: str, dataset_root: Path) -> Bucket:
    try:
        name, remainder = specification.split("=", 1)
        relative, dimensions = remainder.rsplit("@", 1)
        width_text, height_text = dimensions.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "Buckets must use NAME=RELATIVE_DIRECTORY@WIDTHxHEIGHT"
        ) from error
    directory = (dataset_root / relative).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset bucket does not exist: {directory}")
    if width % 16 or height % 16:
        raise ValueError(f"Bucket {name!r} dimensions must be divisible by 16")
    return Bucket(name=name, directory=directory, width=width, height=height)


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(bucket: Bucket) -> list[Path]:
    matches = []
    for path in sorted(bucket.directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as image:
                size = ImageOps.exif_transpose(image).size
        except (OSError, ValueError):
            continue
        if size == (bucket.width, bucket.height):
            matches.append(path)
    return matches


def split_bucket(
    bucket: Bucket,
    *,
    train_count: int,
    validation_count: int,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    paths = image_paths(bucket)
    required = train_count + validation_count
    if len(paths) < required:
        raise RuntimeError(
            f"Bucket {bucket.name!r} has {len(paths)} matching images; "
            f"{required} are required"
        )
    rng = random.Random(f"{seed}:{bucket.name}")
    rng.shuffle(paths)
    return paths[:train_count], paths[train_count:required]


def load_pixels(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def regression_rows(
    pixels: torch.Tensor, latents: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    target = pixels.movedim(-1, 1)
    target = F.interpolate(
        target,
        size=latents.shape[-2:],
        mode="area",
    )
    target = target.mul(2.0).sub(1.0)
    x = latents.movedim(1, -1).reshape(-1, latents.shape[1]).double().cpu()
    y = target.movedim(1, -1).reshape(-1, 3).double().cpu()
    ones = torch.ones((x.shape[0], 1), dtype=torch.float64)
    return torch.cat((x, ones), dim=1), y


def save_resume_state(
    path: Path,
    *,
    xtx: torch.Tensor,
    xty: torch.Tensor,
    y_sum: torch.Tensor,
    rows: int,
    processed: list[str],
    settings: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            xtx=xtx.numpy(),
            xty=xty.numpy(),
            y_sum=y_sum.numpy(),
            rows=np.asarray(rows, dtype=np.int64),
        )
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps({"settings": settings, "processed": processed}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_resume_state(
    path: Path, settings: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list[str]]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.exists() or not metadata_path.exists():
        return (
            torch.zeros((129, 129), dtype=torch.float64),
            torch.zeros((129, 3), dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            0,
            [],
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("settings") != settings:
        raise RuntimeError(
            f"Resume settings do not match {metadata_path}; choose another state file"
        )
    with np.load(path) as state:
        return (
            torch.from_numpy(state["xtx"]).double(),
            torch.from_numpy(state["xty"]).double(),
            torch.from_numpy(state["y_sum"]).double(),
            int(state["rows"]),
            list(metadata["processed"]),
        )


def solve_factors(
    xtx: torch.Tensor, xty: torch.Tensor, rows: int, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    covariance = xtx / float(rows)
    cross = xty / float(rows)
    penalty = torch.eye(129, dtype=torch.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    solution = torch.linalg.solve(covariance + penalty, cross)
    return solution[:-1].float(), solution[-1].float()


def predict_grid(
    latents: torch.Tensor, factors: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return F.linear(
        latents.movedim(1, -1).float().cpu(),
        factors.transpose(0, 1),
        bias,
    )


def validation_metrics(
    vae,
    validation: dict[str, list[Path]],
    factors: torch.Tensor,
    bias: torch.Tensor,
    training_mean: torch.Tensor,
    contact_sheet: Path,
) -> dict:
    totals: dict[str, dict[str, object]] = {}
    previews: list[Image.Image] = []
    for bucket_name, paths in validation.items():
        square_error = 0.0
        baseline_error = 0.0
        absolute_error = 0.0
        channel_error = torch.zeros(3, dtype=torch.float64)
        values = 0
        for path in paths:
            pixels = load_pixels(path)
            latents = vae.encode(pixels)
            prediction = predict_grid(latents, factors, bias)
            target = F.interpolate(
                pixels.movedim(-1, 1),
                size=latents.shape[-2:],
                mode="area",
            ).movedim(1, -1).mul(2.0).sub(1.0)
            difference = prediction.double() - target.double()
            baseline = training_mean.view(1, 1, 1, 3) - target.double()
            square_error += float(difference.square().sum())
            baseline_error += float(baseline.square().sum())
            absolute_error += float(difference.abs().sum())
            channel_error += difference.sum(dim=(0, 1, 2))
            values += difference.numel()
            if len(previews) < 8:
                preview = prediction[0].movedim(-1, 0).unsqueeze(0)
                preview = F.interpolate(
                    preview,
                    size=(pixels.shape[1], pixels.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                )[0].movedim(0, -1)
                preview = preview.add(1.0).div(2.0).clamp(0, 1)
                preview_image = Image.fromarray(
                    preview.mul(255).byte().numpy(), mode="RGB"
                )
                original = Image.fromarray(
                    pixels[0].mul(255).byte().numpy(), mode="RGB"
                )
                canvas = Image.new(
                    "RGB",
                    (original.width * 2, original.height),
                )
                canvas.paste(original, (0, 0))
                canvas.paste(preview_image, (original.width, 0))
                previews.append(canvas)
        rmse = math.sqrt(square_error / values)
        baseline_rmse = math.sqrt(baseline_error / values)
        totals[bucket_name] = {
            "mae": absolute_error / values,
            "rmse": rmse,
            "psnr": 20.0 * math.log10(2.0 / max(rmse, 1.0e-12)),
            "baseline_rmse": baseline_rmse,
            "psnr_gain_db": 20.0 * math.log10(
                max(baseline_rmse, 1.0e-12) / max(rmse, 1.0e-12)
            ),
            "channel_bias": (channel_error / (values // 3)).tolist(),
            "images": len(paths),
        }
    if previews:
        width = max(image.width for image in previews)
        height = sum(image.height for image in previews)
        sheet = Image.new("RGB", (width, height))
        offset = 0
        for image in previews:
            sheet.paste(image, (0, offset))
            offset += image.height
        contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(contact_sheet, quality=90)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vae-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument(
        "--bucket",
        action="append",
        help=(
            "NAME=RELATIVE_DIRECTORY@WIDTHxHEIGHT; repeat to replace the "
            "built-in balanced bucket set"
        ),
    )
    parser.add_argument("--train-per-bucket", type=int, default=256)
    parser.add_argument("--validation-per-bucket", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260730)
    arguments = parser.parse_args()

    dataset_root = arguments.dataset_root.resolve()
    bucket_specs = arguments.bucket or DEFAULT_BUCKETS
    buckets = [parse_bucket(item, dataset_root) for item in bucket_specs]
    vae_path = Path(folder_paths.get_full_path_or_raise("vae", arguments.vae_name))
    digest = checkpoint_hash(vae_path)
    settings = {
        "checkpoint_sha256": digest,
        "buckets": [
            {
                "name": bucket.name,
                "directory": str(bucket.directory.relative_to(dataset_root)),
                "width": bucket.width,
                "height": bucket.height,
            }
            for bucket in buckets
        ],
        "train_per_bucket": arguments.train_per_bucket,
        "validation_per_bucket": arguments.validation_per_bucket,
        "ridge": arguments.ridge,
        "seed": arguments.seed,
        "rgb_alignment": "torch_area_patch16",
        "rgb_range": "[-1,1]",
    }
    state_path = arguments.state or arguments.output.with_suffix(".state.npz")
    xtx, xty, y_sum, rows, processed = load_resume_state(state_path, settings)
    processed_set = set(processed)

    training: dict[str, list[Path]] = {}
    validation: dict[str, list[Path]] = {}
    for bucket in buckets:
        training[bucket.name], validation[bucket.name] = split_bucket(
            bucket,
            train_count=arguments.train_per_bucket,
            validation_count=arguments.validation_per_bucket,
            seed=arguments.seed,
        )

    state = comfy.utils.load_torch_file(str(vae_path), safe_load=True)
    vae = load_dinac(state)
    for bucket_name, paths in training.items():
        for path in paths:
            identity = f"{bucket_name}:{path.relative_to(dataset_root)}"
            if identity in processed_set:
                continue
            pixels = load_pixels(path)
            latents = vae.encode(pixels)
            x, y = regression_rows(pixels, latents)
            xtx += x.transpose(0, 1) @ x
            xty += x.transpose(0, 1) @ y
            y_sum += y.sum(dim=0)
            rows += x.shape[0]
            processed.append(identity)
            processed_set.add(identity)
            save_resume_state(
                state_path,
                xtx=xtx,
                xty=xty,
                y_sum=y_sum,
                rows=rows,
                processed=processed,
                settings=settings,
            )
            print(f"encoded {len(processed)}/{len(buckets) * arguments.train_per_bucket}")

    factors, bias = solve_factors(xtx, xty, rows, arguments.ridge)
    training_mean = (y_sum / rows).float()
    contact_sheet = arguments.output.with_suffix(".contacts.jpg")
    metrics = validation_metrics(
        vae,
        validation,
        factors,
        bias,
        training_mean,
        contact_sheet,
    )
    result = {
        "format": "comfy_latent_rgb_affine_v1",
        "settings": settings,
        "rows": rows,
        "latent_rgb_factors": factors.tolist(),
        "latent_rgb_factors_bias": bias.tolist(),
        "validation": metrics,
        "contact_sheet": str(contact_sheet),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
