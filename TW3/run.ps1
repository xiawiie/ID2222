param(
    [string]$Python = "python",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExperimentArgs
)

$ErrorActionPreference = "Stop"
& $Python (Join-Path $PSScriptRoot "main.py") @ExperimentArgs
if ($LASTEXITCODE -ne 0) {
    throw "TW3 experiment failed (exit $LASTEXITCODE)."
}
