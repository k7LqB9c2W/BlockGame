from __future__ import annotations

import argparse
import csv
import pathlib
import struct
from dataclasses import dataclass


@dataclass
class CaptureScore:
    path: pathlib.Path
    darkest_band_mean: float
    worst_row: int


def read_bmp_rgba(path: pathlib.Path) -> tuple[int, int, list[list[tuple[int, int, int, int]]]]:
    data = path.read_bytes()
    if data[0:2] != b"BM":
        raise ValueError(f"{path} is not a BMP file")

    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    header_size = struct.unpack_from("<I", data, 14)[0]
    if header_size < 40:
        raise ValueError(f"{path} has unsupported BMP header")

    width = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    bit_count = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    if bit_count != 32 or compression != 0:
        raise ValueError(f"{path} is not an uncompressed 32-bit BMP")

    width = abs(width)
    row_count = abs(height)
    row_stride = width * 4
    pixels: list[list[tuple[int, int, int, int]]] = []

    for row_index in range(row_count):
        start = pixel_offset + row_index * row_stride
        row: list[tuple[int, int, int, int]] = []
        for pixel_index in range(width):
            b, g, r, a = data[start + pixel_index * 4 : start + pixel_index * 4 + 4]
            row.append((r, g, b, a))
        pixels.append(row)

    pixels.reverse()
    return width, row_count, pixels


def luminance(pixel: tuple[int, int, int, int]) -> float:
    r, g, b, _ = pixel
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def score_capture(path: pathlib.Path) -> CaptureScore:
    width, height, rows = read_bmp_rgba(path)
    row_start = int(height * 0.12)
    row_end = int(height * 0.82)
    col_start = int(width * 0.20)
    col_end = int(width * 0.80)

    best_score = float("inf")
    best_row = row_start
    for row_index in range(row_start, row_end):
        values = [luminance(pixel) for pixel in rows[row_index][col_start:col_end]]
        if not values:
            continue
        values.sort()
        sample_count = max(12, len(values) // 8)
        darkest_mean = sum(values[:sample_count]) / sample_count
        if darkest_mean < best_score:
            best_score = darkest_mean
            best_row = row_index

    return CaptureScore(path=path, darkest_band_mean=best_score, worst_row=best_row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score screenshot sweep captures for dark horizon bands.")
    parser.add_argument("capture_dir", type=pathlib.Path)
    args = parser.parse_args()

    capture_dir = args.capture_dir.resolve()
    bmp_files = sorted(capture_dir.glob("*.bmp"))
    if not bmp_files:
        print(f"No BMP captures found in {capture_dir}")
        return 0

    scores = [score_capture(path) for path in bmp_files]
    scores.sort(key=lambda item: item.darkest_band_mean)

    analysis_path = capture_dir / "analysis.csv"
    with analysis_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "darkest_band_mean", "worst_row"])
        for score in scores:
            writer.writerow([score.path.name, f"{score.darkest_band_mean:.2f}", score.worst_row])

    print("Worst dark-band candidates:")
    for score in scores[:10]:
        print(f"  {score.path.name}: darkest_band_mean={score.darkest_band_mean:.2f} row={score.worst_row}")
    print(f"Analysis written to {analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
