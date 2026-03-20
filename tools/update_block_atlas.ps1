param(
    [string]$AtlasPath = "block_atlas.png",
    [string[]]$Tiles = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$resolvedAtlasPath = (Resolve-Path $AtlasPath).Path
$atlasDirectory = Split-Path $resolvedAtlasPath -Parent
$tempPath = Join-Path $atlasDirectory "block_atlas.tmp.png"

if (Test-Path $tempPath)
{
    Remove-Item $tempPath -Force
}

if ($Tiles.Count -eq 0)
{
    throw "Provide one or more tile image paths via -Tiles."
}

$existing = [System.Drawing.Bitmap]::FromFile($resolvedAtlasPath)
try
{
    $tileSize = 16
    $bitmap = New-Object System.Drawing.Bitmap -ArgumentList @(
        $existing.Width,
        ($existing.Height + ($Tiles.Count * $tileSize)),
        [System.Drawing.Imaging.PixelFormat]::Format32bppArgb
    )

    try
    {
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        try
        {
            $graphics.Clear([System.Drawing.Color]::Transparent)
            $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::NearestNeighbor
            $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::Half
            $graphics.CompositingMode = [System.Drawing.Drawing2D.CompositingMode]::SourceCopy
            $graphics.DrawImage($existing, 0, 0, $existing.Width, $existing.Height)

            for ($i = 0; $i -lt $Tiles.Count; ++$i)
            {
                $tilePath = (Resolve-Path $Tiles[$i]).Path
                $tile = [System.Drawing.Image]::FromFile($tilePath)
                try
                {
                    $graphics.DrawImage($tile, 0, $existing.Height + ($i * $tileSize), $tileSize, $tileSize)
                }
                finally
                {
                    $tile.Dispose()
                }
            }
        }
        finally
        {
            $graphics.Dispose()
        }

        $bitmap.Save($tempPath, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally
    {
        $bitmap.Dispose()
    }
}
finally
{
    $existing.Dispose()
}

Move-Item -Path $tempPath -Destination $resolvedAtlasPath -Force
