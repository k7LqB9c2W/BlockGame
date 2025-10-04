"""Utility for locating the nearest Little Mountains biome column.

This re-implements the deterministic biome site selection from `ChunkManager`
using the hard-coded world seed found in `main.cpp`. It reports the nearest
column whose Little Mountains weight is non-zero when the player spawns at the
origin (world X/Z = 0).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable, Optional

# Constants mirrored from `chunk_manager.h` / `chunk_manager.cpp`.
CHUNK_SIZE = 16
BIOME_SIZE_IN_CHUNKS = 30
LITTLE_MOUNTAINS_INDEX = 3  # enum order in `BiomeId`
LITTLE_MOUNTAINS_FOOTPRINT = 2.4
MARGIN_RATIO = 0.2
LITTLE_MOUNTAINS_INFLUENCE_CUTOFF = 0.95  # smoothstep upper bound


@dataclasses.dataclass(frozen=True)
class Biome:
    name: str
    footprint_multiplier: float


BIOMES: tuple[Biome, ...] = (
    Biome("Grasslands", 1.0),
    Biome("Forest", 1.0),
    Biome("Desert", 1.0),
    Biome("Little Mountains", LITTLE_MOUNTAINS_FOOTPRINT),
    Biome("Ocean", 1.0),
)


def hash_to_unit_float(x: int, y: int, z: int) -> float:
    """Mimics the integer hash from `chunk_manager.cpp`."""

    h = (x * 374_761_393 + y * 668_265_263 + z * 2_147_483_647) & 0xFFFF_FFFF
    h = (h ^ (h >> 13)) & 0xFFFF_FFFF
    h = (h * 1_274_126_177) & 0xFFFF_FFFF
    h ^= (h >> 16)
    h &= 0xFFFF_FFFF
    return (h & 0xFF_FFFF) / float(0xFF_FFFF)


def biome_index_for_region(region_x: int, region_z: int) -> int:
    selector = hash_to_unit_float(region_x, 31, region_z)
    return min(int(selector * len(BIOMES)), len(BIOMES) - 1)


@dataclasses.dataclass(frozen=True)
class BiomeSite:
    biome_index: int
    center_x: float
    center_z: float
    half_extent_x: float
    half_extent_z: float

    @property
    def radius(self) -> float:
        # Little Mountains footprint yields a circular influence area.
        return min(self.half_extent_x, self.half_extent_z) * LITTLE_MOUNTAINS_INFLUENCE_CUTOFF


def compute_site(region_x: int, region_z: int, biome: Biome) -> BiomeSite:
    scaled_biome_size = BIOME_SIZE_IN_CHUNKS * biome.footprint_multiplier
    region_width = CHUNK_SIZE * scaled_biome_size
    region_depth = CHUNK_SIZE * scaled_biome_size
    margin_x = region_width * MARGIN_RATIO
    margin_z = region_depth * MARGIN_RATIO
    jitter_x = hash_to_unit_float(region_x, 137, region_z)
    jitter_z = hash_to_unit_float(region_x, 613, region_z)
    available_width = max(region_width - 2.0 * margin_x, 0.0)
    available_depth = max(region_depth - 2.0 * margin_z, 0.0)
    base_x = region_x * region_width
    base_z = region_z * region_depth

    center_x = base_x + margin_x + available_width * jitter_x
    center_z = base_z + margin_z + available_depth * jitter_z
    half_extent_x = region_width * 0.5
    half_extent_z = region_depth * 0.5

    return BiomeSite(
        biome_index=biome_index_for_region(region_x, region_z),
        center_x=center_x,
        center_z=center_z,
        half_extent_x=half_extent_x,
        half_extent_z=half_extent_z,
    )


def iter_little_mountains_sites(radius_regions: int = 12) -> Iterable[BiomeSite]:
    for region_z in range(-radius_regions, radius_regions + 1):
        for region_x in range(-radius_regions, radius_regions + 1):
            biome_idx = biome_index_for_region(region_x, region_z)
            if biome_idx != LITTLE_MOUNTAINS_INDEX:
                continue
            biome = BIOMES[biome_idx]
            yield compute_site(region_x, region_z, biome)


def normalized_distance(column_x: int, column_z: int, site: BiomeSite) -> float:
    dx = column_x + 0.5 - site.center_x
    dz = column_z + 0.5 - site.center_z
    return math.hypot(dx / site.half_extent_x, dz / site.half_extent_z)


def find_nearest_little_mountains() -> tuple[tuple[int, int], float, BiomeSite]:
    best_column: Optional[tuple[int, int]] = None
    best_distance: float = math.inf
    best_site: Optional[BiomeSite] = None

    for site in iter_little_mountains_sites():
        center_distance = math.hypot(site.center_x, site.center_z)
        influence_radius = site.radius
        if center_distance <= influence_radius:
            candidate_center_x = 0.0
            candidate_center_z = 0.0
        else:
            scale = 1.0 - influence_radius / center_distance
            candidate_center_x = site.center_x * scale
            candidate_center_z = site.center_z * scale

        base_x = int(round(candidate_center_x - 0.5))
        base_z = int(round(candidate_center_z - 0.5))

        for offset_x in range(-3, 4):
            for offset_z in range(-3, 4):
                column_x = base_x + offset_x
                column_z = base_z + offset_z
                if normalized_distance(column_x, column_z, site) >= LITTLE_MOUNTAINS_INFLUENCE_CUTOFF:
                    continue
                planar_distance = math.hypot(column_x, column_z)
                if planar_distance < best_distance:
                    best_distance = planar_distance
                    best_column = (column_x, column_z)
                    best_site = site

    if best_column is None or best_site is None:
        raise RuntimeError("No Little Mountains columns found within the search radius")

    return best_column, best_distance, best_site


def main() -> None:
    column, distance, site = find_nearest_little_mountains()
    print("Nearest Little Mountains column:")
    print(f"  world column: {column}")
    print(f"  horizontal distance from spawn: {distance:.3f} blocks")
    print("  biome site center:"
          f" ({site.center_x:.3f}, {site.center_z:.3f}) with half extents"
          f" ({site.half_extent_x:.1f}, {site.half_extent_z:.1f})")


if __name__ == "__main__":
    main()
