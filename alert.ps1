param(
    [string]$Task = "Codex finished"
)

$env:NTFY_TOPIC = "codex-done-A8fK29xPq7LmZ4rT"

Start-Process codex -Wait

Invoke-RestMethod -Method POST `
  -Body $Task `
  "https://ntfy.sh/$env:NTFY_TOPIC"