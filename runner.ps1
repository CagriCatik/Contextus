# runner.ps1
# Models:
# - deepseek-r1:1.5b
# - qwen3:1.7b

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

$models = @(
    "deepseek-r1:1.5b",
    "qwen3:1.7b",
    "glm-4.6:cloud",
    "gpt-oss:20b",
    "gpt-oss:20b-cloud",
    "gpt-oss:120b-cloud"
)

Write-Host "Select a model:"
for ($i = 0; $i -lt $models.Count; $i++) {
    Write-Host "[$($i+1)] $($models[$i])"
}

$choice = Read-Host "Enter number"
$index = [int]$choice - 1

if ($index -lt 0 -or $index -ge $models.Count) {
    Write-Host "Invalid selection."
    exit 1
}

$Model = $models[$index]

Write-Host "Installing model: $Model"
ollama pull $Model

Write-Host "Running chat_cli.py with $Model"
python "$scriptDir/chat_cli.py" --model $Model --top-k 6 --max-context-chars 3500
