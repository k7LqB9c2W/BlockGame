# Mob Assets

Put Bedrock-style geometry JSON files in this folder.

Current BlockGame mob import rules:
- Files are discovered from `assets/mobs/*.json`.
- Legacy Bedrock geometry roots like `geometry.pig.v1.8` are supported.
- Modern `minecraft:geometry` files are also accepted if they use a standard `description` plus `bones` layout.
- Box UV cubes are supported.
- The loader looks for a texture next to the JSON using the same base name.
  - Example: `pig.geo.json` will try `pig.png`.
- If the texture is missing, the mob renders with a default pink fallback material.
- Per-bone geometry is preserved as baked parts so simple runtime animation can be applied without skeletal skinning.
- Pig-style four-leg walk animation currently expects Bedrock leg bone names `leg0`, `leg1`, `leg2`, and `leg3`.

For the current test pass, BlockGame supports simple passive AI plus a lightweight walk-cycle animation, not full Bedrock animation controllers.
