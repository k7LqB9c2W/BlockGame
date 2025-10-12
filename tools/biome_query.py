"""Interactive biome query utility for BlockGame terrain."""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError as exc:  # pragma: no cover - Python <3.11 fallback
    raise SystemExit("Python 3.11 or newer is required to run biome_query.py") from exc


# ---------------------------------------------------------------------------
# Data model


PROPERTY_BITS: dict[str, int] = {
    "hot": 1 << 0,
    "temperate": 1 << 1,
    "cold": 1 << 2,
    "inland": 1 << 3,
    "land": 1 << 4,
    "ocean": 1 << 5,
    "wet": 1 << 6,
    "neutral": 1 << 7,
    "neither_wet_nor_dry": 1 << 7,
    "neutral_hydration": 1 << 7,
    "dry": 1 << 8,
    "barren": 1 << 9,
    "balanced": 1 << 10,
    "overgrown": 1 << 11,
    "mountain": 1 << 12,
    "low_terrain": 1 << 13,
    "antimountain": 1 << 14,
    "anti_mountain": 1 << 14,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hash_combine(a: int, b: int) -> int:
    mask = 0xFFFFFFFFFFFFFFFF
    a &= mask
    b &= mask
    result = (a ^ (b + 0x9E3779B97F4A7C15 + ((a << 6) & mask) + (a >> 2))) & mask
    return result


def _floor_div(value: int, divisor: int) -> int:
    quotient = value // divisor
    remainder = value % divisor
    if remainder and ((remainder < 0) != (divisor < 0)):
        quotient -= 1
    return quotient


def _length_squared(ax: int, ay: int, bx: int, by: int) -> float:
    dx = ax - bx
    dz = ay - by
    return float(dx * dx + dz * dz)


def _smooth_step(t: float) -> float:
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _evaluate_curve(t: float, curve: str) -> float:
    t = _clamp(t, 0.0, 1.0)
    if curve == "step":
        return 1.0 if t >= 0.5 else 0.0
    if curve == "linear":
        return t
    # Square / smooth fall-back
    if t < 0.5:
        return _clamp(2.0 * t * t, 0.0, 1.0)
    inv = 1.0 - t
    return _clamp(1.0 - 2.0 * inv * inv, 0.0, 1.0)


class _Random:
    """Xorshift-inspired RNG mirroring the game's implementation."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFFFFFFFFFF

    def next_uint32(self) -> int:
        state = self._state
        state ^= (state >> 12) & 0xFFFFFFFFFFFFFFFF
        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
        state ^= (state >> 27) & 0xFFFFFFFFFFFFFFFF
        self._state = state
        return (state * 2685821657736338717 & 0xFFFFFFFFFFFFFFFF) >> 32

    def next_float(self) -> float:
        return self.next_uint32() / 0xFFFFFFFF

    def next_float_signed(self) -> float:
        return self.next_float() * 2.0 - 1.0

    def next_int(self, minimum: int, maximum: int) -> int:
        if maximum <= minimum:
            return minimum
        span = maximum - minimum + 1
        return minimum + int(self.next_uint32() % span)


@dataclasses.dataclass
class TransitionDefinition:
    biome_id: str
    chance: float
    width: int
    property_mask: int
    biome: Optional["BiomeDefinition"] = None


@dataclasses.dataclass
class SubBiomeDefinition:
    biome_id: str
    chance: float
    min_radius: float
    max_radius: float
    biome: Optional["BiomeDefinition"] = None

    def sample_radius(self, default_radius: float, noise: float) -> float:
        low = self.min_radius if self.min_radius > 0.0 else default_radius * 0.25
        high = self.max_radius if self.max_radius > 0.0 else default_radius * 0.75
        low = max(low, 1.0)
        high = max(high, low)
        t = _clamp(noise, 0.0, 1.0)
        return low + (high - low) * t


@dataclasses.dataclass
class BiomeDefinition:
    id: str
    name: str
    radius: float
    radius_variation: float
    spawn_chance: float
    footprint_multiplier: float
    interpolation_curve: str
    interpolation_weight: float
    fixed_radius: bool
    flags: Tuple[str, ...]
    property_bits: int
    sub_biomes: Tuple[SubBiomeDefinition, ...]
    max_sub_biome_count: float
    transition_biomes: Tuple[TransitionDefinition, ...]

    def __post_init__(self) -> None:
        self.flags = tuple(flag.lower() for flag in self.flags)
        self._flag_lookup = frozenset(self.flags)
        self._is_ocean = "ocean" in self._flag_lookup

    def has_flag(self, flag: str) -> bool:
        return flag.lower() in self._flag_lookup

    def has_property(self, name: str) -> bool:
        bit = PROPERTY_BITS.get(name.lower())
        if bit is None:
            return False
        return (self.property_bits & bit) == bit

    @property
    def is_ocean(self) -> bool:
        return self._is_ocean

    @property
    def max_radius(self) -> float:
        return max(self.radius + self.radius_variation, self.radius, 1.0)

    @property
    def min_radius(self) -> float:
        return max(self.radius - self.radius_variation, 1.0)


def _parse_properties(values: Sequence[str]) -> int:
    bits = 0
    for value in values:
        bit = PROPERTY_BITS.get(value.lower())
        if bit is not None:
            bits |= bit
    return bits


def _parse_biome(path: Path) -> BiomeDefinition:
    data = tomllib.loads(path.read_text())

    def get_float(key: str, default: float) -> float:
        value = data.get(key, default)
        if isinstance(value, (int, float)):
            return float(value)
        return default

    def get_int(key: str, default: int) -> int:
        value = data.get(key, default)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return default

    flags = tuple(data.get("flags", ()) or ())
    properties = tuple(data.get("properties", ()) or ())

    def parse_sub_biomes() -> Tuple[SubBiomeDefinition, ...]:
        entries = data.get("sub_biomes", []) or []
        result: list[SubBiomeDefinition] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            result.append(
                SubBiomeDefinition(
                    biome_id=str(entry.get("id", "")),
                    chance=float(entry.get("chance", 0.0)),
                    min_radius=float(entry.get("min_radius", 0.0)),
                    max_radius=float(entry.get("max_radius", 0.0)),
                )
            )
        return tuple(result)

    def parse_transitions() -> Tuple[TransitionDefinition, ...]:
        entries = data.get("transition_biomes", []) or []
        result: list[TransitionDefinition] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            mask = _parse_properties(entry.get("properties", []) or [])
            result.append(
                TransitionDefinition(
                    biome_id=str(entry.get("id", "")),
                    chance=float(entry.get("chance", 0.0)),
                    width=int(entry.get("width", 0)),
                    property_mask=mask,
                )
            )
        return tuple(result)

    radius = get_float("radius", 256.0)
    radius_variation = get_float("radius_variation", 0.0)
    radius = max(radius, 1.0)
    radius_variation = max(radius_variation, 0.0)

    interpolation_curve = str(data.get("interpolation_curve", "square")).lower()
    if interpolation_curve not in {"square", "linear", "step"}:
        interpolation_curve = "square"

    footprint = get_float("footprint_multiplier", 1.0)
    footprint = _clamp(footprint, 0.25, 3.0)

    return BiomeDefinition(
        id=str(data.get("id", path.stem)).lower(),
        name=str(data.get("name", path.stem)),
        radius=radius,
        radius_variation=radius_variation,
        spawn_chance=get_float("spawn_chance", 1.0),
        footprint_multiplier=footprint,
        interpolation_curve=interpolation_curve,
        interpolation_weight=get_float("interpolation_weight", 1.0),
        fixed_radius=bool(data.get("fixed_radius", False)),
        flags=flags,
        property_bits=_parse_properties(properties),
        sub_biomes=parse_sub_biomes(),
        max_sub_biome_count=get_float("max_sub_biome_count", 0.0),
        transition_biomes=parse_transitions(),
    )


class BiomeDatabase:
    def __init__(self, directory: Path) -> None:
        files = sorted(p for p in directory.glob("*.toml") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No biome definitions found in {directory}")

        self.definitions: list[BiomeDefinition] = [_parse_biome(path) for path in files]
        self.by_id = {definition.id: definition for definition in self.definitions}
        self.by_name = {definition.name.lower(): definition for definition in self.definitions}

        for definition in self.definitions:
            linked_subs: list[SubBiomeDefinition] = []
            for sub in definition.sub_biomes:
                target = self.by_id.get(sub.biome_id.lower())
                if target is None:
                    continue
                sub.biome = target
                linked_subs.append(sub)
            definition.sub_biomes = tuple(linked_subs)

            linked_transitions: list[TransitionDefinition] = []
            for transition in definition.transition_biomes:
                target = self.by_id.get(transition.biome_id.lower())
                if target is None:
                    continue
                transition.biome = target
                linked_transitions.append(transition)
            definition.transition_biomes = tuple(linked_transitions)

        self.max_biome_radius = max(definition.max_radius for definition in self.definitions)

    def get(self, name_or_id: str) -> Optional[BiomeDefinition]:
        key = name_or_id.lower()
        return self.by_id.get(key) or self.by_name.get(key)


@dataclasses.dataclass
class BiomeSeed:
    biome: BiomeDefinition
    position: Tuple[int, int]
    radius: float
    weight: float


@dataclasses.dataclass
class ChunkSeeds:
    seeds: Tuple[BiomeSeed, ...]
    max_radius: int


@dataclasses.dataclass
class BiomeContribution:
    biome: BiomeDefinition
    weight: float
    normalized_distance: float
    distance: float
    radius: float


@dataclasses.dataclass
class ClimateSample:
    contributions: Tuple[BiomeContribution, ...]

    @property
    def dominant(self) -> Optional[BiomeContribution]:
        return self.contributions[0] if self.contributions else None


class ClimateModel:
    def __init__(self, database: BiomeDatabase, seed: int) -> None:
        self.database = database
        self.seed = seed & 0xFFFFFFFFFFFFFFFF
        self.chunk_span = max(64, int(math.ceil(database.max_biome_radius * 1.75)))
        alignment = 32
        self.chunk_span = max(alignment, ((self.chunk_span + alignment - 1) // alignment) * alignment)
        self.neighbor_radius = max(2, int(math.ceil(database.max_biome_radius / self.chunk_span)) + 1)

        self._chunk_cache: dict[Tuple[int, int], ChunkSeeds] = {}

        self._biome_selection: list[BiomeDefinition] = []
        self._weight_prefix: list[float] = []
        total = 0.0
        for definition in database.definitions:
            if definition.spawn_chance <= 0.0:
                continue
            radius_scale = max(definition.radius, 1.0)
            weight = max(definition.spawn_chance * definition.footprint_multiplier, 0.0)
            weight /= max(radius_scale, 1.0)
            if definition.has_property("ocean"):
                weight *= 1.25
            if definition.has_property("mountain"):
                weight *= 0.85
            if definition.has_property("low_terrain"):
                weight *= 1.1
            if weight <= 0.0:
                continue
            total += weight
            self._biome_selection.append(definition)
            self._weight_prefix.append(total)

        if not self._biome_selection:
            raise RuntimeError("Biome selection table is empty")

    # --------------------------- chunk / seed generation -----------------

    def _chunk_key(self, chunk_x: int, chunk_z: int) -> Tuple[int, int]:
        return (chunk_x, chunk_z)

    def _chunk_seeds(self, chunk_x: int, chunk_z: int) -> ChunkSeeds:
        key = self._chunk_key(chunk_x, chunk_z)
        cached = self._chunk_cache.get(key)
        if cached is not None:
            return cached
        built = self._build_chunk_seeds(chunk_x, chunk_z)
        self._chunk_cache[key] = built
        return built

    def _build_chunk_seeds(self, chunk_x: int, chunk_z: int) -> ChunkSeeds:
        base_x = chunk_x * self.chunk_span
        base_z = chunk_z * self.chunk_span

        seed_value = self.seed
        seed_value = _hash_combine(seed_value, (chunk_x * 73856093) & 0xFFFFFFFFFFFFFFFF)
        seed_value = _hash_combine(seed_value, (chunk_z * 19349663) & 0xFFFFFFFFFFFFFFFF)
        rng = _Random(seed_value)

        seeds: list[BiomeSeed] = []
        max_radius = 0
        rejections = 0
        max_seeds = 48
        max_rejections = 96

        while len(seeds) < max_seeds and rejections < max_rejections:
            world_x = base_x + rng.next_int(0, self.chunk_span - 1)
            world_z = base_z + rng.next_int(0, self.chunk_span - 1)
            seed = self._create_seed(rng, world_x, world_z)
            if seed is None:
                rejections += 1
                continue
            if not self._is_valid_placement(seed.position, seed.radius, seeds):
                rejections += 1
                continue
            seeds.append(seed)
            max_radius = max(max_radius, int(math.ceil(seed.radius)))
            rejections = 0
            before = len(seeds)
            self._spawn_sub_biomes(seed, seeds, rng)
            for new_seed in seeds[before:]:
                max_radius = max(max_radius, int(math.ceil(new_seed.radius)))

        if not seeds:
            fallback = self._create_seed(rng, base_x + self.chunk_span // 2, base_z + self.chunk_span // 2)
            if fallback is not None:
                seeds.append(fallback)
                max_radius = max(max_radius, int(math.ceil(fallback.radius)))
                before = len(seeds)
                self._spawn_sub_biomes(fallback, seeds, rng)
                for new_seed in seeds[before:]:
                    max_radius = max(max_radius, int(math.ceil(new_seed.radius)))

        return ChunkSeeds(tuple(seeds), max_radius)

    def _create_seed(self, rng: _Random, world_x: int, world_z: int) -> Optional[BiomeSeed]:
        biome = self._choose_biome(rng)
        if biome is None:
            return None
        if biome.fixed_radius or biome.is_ocean:
            radius = biome.radius
        else:
            radius = biome.radius + biome.radius_variation * rng.next_float_signed()
            radius = _clamp(radius, biome.min_radius, biome.max_radius)
        radius = max(radius, 1.0)
        weight = 1.0 / max(radius * math.sqrt(math.pi), 1.0)
        return BiomeSeed(biome=biome, position=(world_x, world_z), radius=radius, weight=weight)

    def _choose_biome(self, rng: _Random) -> Optional[BiomeDefinition]:
        if not self._biome_selection:
            return None
        pick = rng.next_float() * self._weight_prefix[-1]
        lo, hi = 0, len(self._weight_prefix) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if pick <= self._weight_prefix[mid]:
                hi = mid
            else:
                lo = mid + 1
        return self._biome_selection[lo]

    def _is_valid_placement(self, position: Tuple[int, int], radius: float, seeds: Sequence[BiomeSeed]) -> bool:
        for other in seeds:
            largest_radius = max(radius, other.radius)
            spacing = _clamp(0.85 - 0.0005 * largest_radius, 0.6, 0.85)
            combined = (radius + other.radius) * spacing
            dist_sq = _length_squared(position[0], position[1], other.position[0], other.position[1])
            if dist_sq < combined * combined:
                return False
        return True

    def _random_in_unit_circle(self, rng: _Random) -> Tuple[float, float]:
        while True:
            x = rng.next_float_signed()
            y = rng.next_float_signed()
            if x * x + y * y <= 1.0:
                return x, y

    def _spawn_sub_biomes(self, parent: BiomeSeed, seeds: list[BiomeSeed], rng: _Random) -> None:
        parent_def = parent.biome
        if not parent_def.sub_biomes:
            return
        max_count = int(math.ceil(parent_def.max_sub_biome_count)) if parent_def.max_sub_biome_count > 0.0 else sys.maxsize
        spawned = 0
        for sub in parent_def.sub_biomes:
            target = sub.biome
            if target is None:
                continue
            if spawned >= max_count:
                break
            probability = _clamp(sub.chance, 0.0, 1.0)
            if probability <= 0.0:
                continue
            if rng.next_float() > probability:
                continue
            offset_x, offset_z = self._random_in_unit_circle(rng)
            parent_radius = max(parent.radius, 1.0)
            distance = parent_radius * 0.6 * math.sqrt(rng.next_float())
            candidate_x = parent.position[0] + int(offset_x * distance)
            candidate_z = parent.position[1] + int(offset_z * distance)
            radius_noise = rng.next_float()
            radius = sub.sample_radius(parent_radius * 0.75, radius_noise)
            radius = _clamp(radius, 4.0, parent_radius)
            candidate_seed = BiomeSeed(
                biome=target,
                position=(candidate_x, candidate_z),
                radius=radius,
                weight=1.0 / max(radius * math.sqrt(math.pi), 1.0),
            )
            if not self._is_valid_placement(candidate_seed.position, candidate_seed.radius, seeds):
                continue
            seeds.append(candidate_seed)
            spawned += 1

    # --------------------------- sampling --------------------------------

    def _gather_candidates(self, world_x: int, world_z: int) -> Iterable[BiomeSeed]:
        chunk_x = _floor_div(world_x, self.chunk_span)
        chunk_z = _floor_div(world_z, self.chunk_span)
        radius = self.neighbor_radius
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                chunk = self._chunk_seeds(chunk_x + dx, chunk_z + dz)
                yield from chunk.seeds

    def sample(self, world_x: float, world_z: float) -> ClimateSample:
        candidates: list[Tuple[BiomeSeed, float, float, float]] = []
        for seed in self._gather_candidates(int(math.floor(world_x)), int(math.floor(world_z))):
            dx = world_x - seed.position[0]
            dz = world_z - seed.position[1]
            distance = math.hypot(dx, dz)
            radius = max(seed.radius, 1.0)
            normalized = distance / radius
            blended = _clamp(1.0 - normalized, 0.0, 1.0)
            influence = _smooth_step(blended)
            if influence <= 1e-6:
                continue
            weight = influence * _evaluate_curve(1.0 - normalized, seed.biome.interpolation_curve)
            weight *= max(seed.biome.interpolation_weight, 0.0)
            if weight <= 1e-6:
                continue
            candidates.append((seed, weight, normalized, distance))

        if not candidates:
            return ClimateSample(())

        candidates.sort(key=lambda entry: entry[1], reverse=True)
        total_weight = sum(weight for _, weight, _, _ in candidates)
        if total_weight <= 1e-6:
            total_weight = 1.0

        contributions: List[BiomeContribution] = []
        for seed, weight, normalized, distance in candidates[:8]:
            contributions.append(
                BiomeContribution(
                    biome=seed.biome,
                    weight=weight / total_weight,
                    normalized_distance=normalized,
                    distance=distance,
                    radius=seed.radius,
                )
            )
        return ClimateSample(tuple(contributions))

    # --------------------------- nearest search ---------------------------

    @dataclasses.dataclass
    class NearestResult:
        biome: BiomeDefinition
        nearest_point: Tuple[float, float]
        distance: float
        seed_center: Tuple[int, int]
        radius: float
        sample: ClimateSample

    def find_nearest(self, world_x: float, world_z: float, target: BiomeDefinition, max_chunk_radius: int = 12) -> Optional["ClimateModel.NearestResult"]:
        chunk_x = _floor_div(int(math.floor(world_x)), self.chunk_span)
        chunk_z = _floor_div(int(math.floor(world_z)), self.chunk_span)
        best: Optional[ClimateModel.NearestResult] = None

        for radius in range(max(self.neighbor_radius, 2), max_chunk_radius + 1):
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    chunk = self._chunk_seeds(chunk_x + dx, chunk_z + dz)
                    for seed in chunk.seeds:
                        if seed.biome is not target:
                            continue
                        candidate = self._nearest_point_on_seed(seed, world_x, world_z)
                        if candidate is None:
                            continue
                        nearest_point, _ = candidate
                        confirmed = self._confirm_destination(seed, world_x, world_z, target, nearest_point)
                        if confirmed is None:
                            continue
                        point, distance, sample = confirmed
                        if best is None or distance < best.distance:
                            best = ClimateModel.NearestResult(
                                biome=target,
                                nearest_point=point,
                                distance=distance,
                                seed_center=seed.position,
                                radius=seed.radius,
                                sample=sample,
                            )
            if best is not None:
                # Early exit once we have a candidate within the current radius.
                break
        return best

    def _nearest_point_on_seed(self, seed: BiomeSeed, world_x: float, world_z: float) -> Optional[Tuple[Tuple[float, float], float]]:
        cx, cz = seed.position
        dx = world_x - cx
        dz = world_z - cz
        distance = math.hypot(dx, dz)
        radius = max(seed.radius, 1.0)
        if distance <= radius:
            return (world_x, world_z), 0.0
        if distance <= 1e-6:
            return (cx, cz), radius
        scale = radius / distance
        nearest_x = cx + dx * scale
        nearest_z = cz + dz * scale
        travel = math.hypot(nearest_x - world_x, nearest_z - world_z)
        return (nearest_x, nearest_z), travel

    def _confirm_destination(
        self,
        seed: BiomeSeed,
        player_x: float,
        player_z: float,
        target: BiomeDefinition,
        initial_point: Tuple[float, float],
    ) -> Optional[Tuple[Tuple[float, float], float, ClimateSample]]:
        cx, cz = seed.position
        test_points: list[Tuple[float, float]] = [initial_point]
        for factor in (0.9, 0.75, 0.6, 0.45, 0.3):
            px = cx + (initial_point[0] - cx) * factor
            pz = cz + (initial_point[1] - cz) * factor
            test_points.append((px, pz))

        for px, pz in test_points:
            sample = self.sample(px, pz)
            dominant = sample.dominant
            if dominant and dominant.biome is target:
                distance = math.hypot(px - player_x, pz - player_z)
                return (px, pz), distance, sample
        return None


# ---------------------------------------------------------------------------
# Worldgen profile helpers


@dataclasses.dataclass
class WorldgenProfile:
    seed: int

    @classmethod
    def load(cls, path: Path) -> "WorldgenProfile":
        if not path.exists():
            return cls(seed=2025)
        data = tomllib.loads(path.read_text())
        seed_value = data.get("seed", 2025)
        if not isinstance(seed_value, int):
            seed_value = 2025
        return cls(seed=seed_value)


# ---------------------------------------------------------------------------
# CLI / interaction helpers


def _format_contributions(sample: ClimateSample) -> str:
    lines: list[str] = []
    dominant = sample.dominant
    if dominant is None:
        return "No biome influences found at this location."
    lines.append(
        f"Dominant biome: {dominant.biome.name} (id '{dominant.biome.id}')\n"
        f"  weight: {dominant.weight * 100:.1f}%\n"
        f"  distance to seed center: {dominant.distance:.1f} blocks\n"
        f"  normalized distance: {dominant.normalized_distance:.3f}"
    )
    others = sample.contributions[1:]
    if others:
        lines.append("Nearby influences:")
        for contrib in others:
            lines.append(
                f"  - {contrib.biome.name} (id '{contrib.biome.id}')"
                f" — {contrib.weight * 100:.1f}% influence, normalized distance {contrib.normalized_distance:.3f}"
            )
    return "\n".join(lines)


def _handle_sample(model: ClimateModel, x: float, z: float) -> None:
    sample = model.sample(x, z)
    print(_format_contributions(sample))


def _handle_nearest(model: ClimateModel, x: float, z: float, biome_name: str) -> None:
    target = model.database.get(biome_name)
    if target is None:
        print(f"Unknown biome '{biome_name}'. Available biomes: {', '.join(sorted(model.database.by_id))}")
        return
    result = model.find_nearest(x, z, target)
    if result is None:
        print(f"No nearby '{target.name}' biome found within the search radius.")
        return
    print(
        f"Nearest {target.name}:\n"
        f"  travel ~{result.distance:.1f} blocks to X={result.nearest_point[0]:.1f}, Z={result.nearest_point[1]:.1f}\n"
        f"  seed center at X={result.seed_center[0]}, Z={result.seed_center[1]} with radius ~{result.radius:.1f}"
    )
    dominant = result.sample.dominant
    if dominant is not None:
        print(
            f"  confirmation: dominant biome at destination is {dominant.biome.name}"
            f" ({dominant.weight * 100:.1f}% influence)"
        )


def _interactive_loop(model: ClimateModel) -> None:
    print("BlockGame biome query utility. Type 'help' for commands, 'quit' to exit.")
    while True:
        try:
            command = input("Command [sample/nearest/help/quit]: ").strip().lower()
        except EOFError:
            print()
            break
        if not command:
            continue
        if command in {"quit", "exit", "q"}:
            break
        if command in {"help", "?"}:
            print(
                "Commands:\n"
                "  sample  - query biome influences at a coordinate.\n"
                "  nearest - locate the nearest biome of a given type.\n"
                "  quit    - exit the tool."
            )
            continue
        if command.startswith("sam"):
            try:
                x = float(input("  X coordinate: "))
                _ = input("  Y coordinate (ignored for biome lookup): ")
                z = float(input("  Z coordinate: "))
            except ValueError:
                print("Invalid coordinate input. Please try again.")
                continue
            _handle_sample(model, x, z)
            continue
        if command.startswith("nea"):
            try:
                x = float(input("  Current X: "))
                _ = input("  Current Y (ignored): ")
                z = float(input("  Current Z: "))
            except ValueError:
                print("Invalid coordinate input. Please try again.")
                continue
            biome_name = input("  Target biome (id or name): ").strip()
            if not biome_name:
                print("Please enter a biome id or name.")
                continue
            _handle_nearest(model, x, z, biome_name)
            continue
        print("Unrecognized command. Type 'help' for usage information.")


def build_model(biomes_path: Path, worldgen_path: Path) -> ClimateModel:
    database = BiomeDatabase(biomes_path)
    profile = WorldgenProfile.load(worldgen_path)
    return ClimateModel(database, profile.seed)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query BlockGame biome data.")
    parser.add_argument("--biomes", type=Path, default=Path("assets/biomes"), help="Path to biome definition directory")
    parser.add_argument("--worldgen", type=Path, default=Path("assets/worldgen.toml"), help="Path to worldgen profile")

    subparsers = parser.add_subparsers(dest="command")

    sample_parser = subparsers.add_parser("sample", help="Query biome influences at a location")
    sample_parser.add_argument("--x", type=float, required=True)
    sample_parser.add_argument("--z", type=float, required=True)

    nearest_parser = subparsers.add_parser("nearest", help="Locate the nearest biome of a given type")
    nearest_parser.add_argument("--x", type=float, required=True)
    nearest_parser.add_argument("--z", type=float, required=True)
    nearest_parser.add_argument("--biome", required=True, help="Biome id or display name")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        model = build_model(args.biomes, args.worldgen)
    except Exception as exc:  # pragma: no cover - surface parsing errors cleanly
        print(f"Failed to initialize biome data: {exc}", file=sys.stderr)
        return 1

    if args.command == "sample":
        _handle_sample(model, args.x, args.z)
        return 0
    if args.command == "nearest":
        _handle_nearest(model, args.x, args.z, args.biome)
        return 0

    _interactive_loop(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

