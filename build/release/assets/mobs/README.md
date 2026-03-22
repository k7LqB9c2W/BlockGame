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

For the current test pass, only static bind-pose geometry is rendered. Animation and AI are not wired yet.
