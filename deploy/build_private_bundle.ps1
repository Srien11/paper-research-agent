[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CorpusDir,
    [string]$RepositoryRoot = "",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = Split-Path -Parent $PSScriptRoot
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $RepositoryRoot "dist-private"
}

$repo = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$corpus = (Resolve-Path -LiteralPath $CorpusDir).Path
$outputParent = [IO.Path]::GetFullPath($OutputDirectory)
$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$bundleName = "paper-research-agent-private-$stamp"
$stageParent = Join-Path $outputParent ".stage-$stamp"
$stageRoot = Join-Path $stageParent $bundleName
$archivePath = Join-Path $outputParent "$bundleName.tar.gz"

function Copy-BundleFile {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required bundle file is missing: $Source"
    }
    $destinationParent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination
}

function Copy-SafeTree {
    param([string]$SourceRoot, [string]$DestinationRoot)
    $sourcePrefix = [IO.Path]::GetFullPath($SourceRoot).TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar
    Get-ChildItem -LiteralPath $SourceRoot -Recurse -File | Where-Object {
        $_.FullName -notmatch "[\\/]__pycache__[\\/]" -and $_.Extension -notin ".pyc", ".pyo"
    } | ForEach-Object {
        $fullName = [IO.Path]::GetFullPath($_.FullName)
        if (-not $fullName.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Bundle source escaped its expected root."
        }
        $relative = $fullName.Substring($sourcePrefix.Length)
        Copy-BundleFile -Source $_.FullName -Destination (Join-Path $DestinationRoot $relative)
    }
}

New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
if (-not $stageParent.StartsWith($outputParent, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Staging directory escaped the requested output directory."
}
New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null

try {
    Copy-BundleFile "$repo\pyproject.toml" "$stageRoot\pyproject.toml"
    Copy-BundleFile "$repo\README.md" "$stageRoot\README.md"
    Copy-SafeTree "$repo\src\paper_research_agent" "$stageRoot\src\paper_research_agent"
    Copy-BundleFile "$repo\scripts\serve_web.py" "$stageRoot\scripts\serve_web.py"
    Copy-SafeTree "$repo\configs\answering" "$stageRoot\configs\answering"
    Copy-SafeTree "$repo\configs\memory" "$stageRoot\configs\memory"
    Copy-SafeTree "$repo\configs\retrieval" "$stageRoot\configs\retrieval"
    Copy-SafeTree "$repo\configs\web" "$stageRoot\configs\web"
    Copy-SafeTree "$repo\deploy" "$stageRoot\deploy"

    $chunks = "$repo\data\processed\chunks\chunks.jsonl"
    $indexDir = "$repo\data\indexes\retrieval-v1"
    $manifestPath = "$indexDir\manifest.json"
    Copy-BundleFile $chunks "$stageRoot\data\processed\chunks\chunks.jsonl"
    Copy-BundleFile $manifestPath "$stageRoot\data\indexes\retrieval-v1\manifest.json"
    Copy-BundleFile "$indexDir\vectors.faiss" "$stageRoot\data\indexes\retrieval-v1\vectors.faiss"
    Copy-BundleFile "$indexDir\metadata.sqlite" "$stageRoot\data\indexes\retrieval-v1\metadata.sqlite"
    Copy-BundleFile "$corpus\core_frozen.jsonl" "$stageRoot\corpus\core_frozen.jsonl"
    Copy-BundleFile "$corpus\challenge_frozen.jsonl" "$stageRoot\corpus\challenge_frozen.jsonl"
    New-Item -ItemType Directory -Force -Path "$stageRoot\data\runtime" | Out-Null

    $manifest = Get-Content -Raw -Encoding UTF8 $manifestPath | ConvertFrom-Json
    $chunkHash = (Get-FileHash -Algorithm SHA256 $chunks).Hash.ToLowerInvariant()
    if ($chunkHash -ne $manifest.chunk_build_sha256) {
        throw "chunks.jsonl does not match the retrieval manifest."
    }
    foreach ($fileName in @("vectors.faiss", "metadata.sqlite")) {
        $actualHash = (Get-FileHash -Algorithm SHA256 (Join-Path $indexDir $fileName)).Hash.ToLowerInvariant()
        $expectedHash = [string]$manifest.files_sha256.$fileName
        if ($actualHash -ne $expectedHash) {
            throw "$fileName does not match the retrieval manifest."
        }
    }

    $forbidden = Get-ChildItem -LiteralPath $stageRoot -Recurse -File | Where-Object {
        $_.Name -like ".env*" -or $_.Extension -in ".pdf", ".pem", ".key" -or
        ($_.FullName -match "[\\/]data[\\/]runtime[\\/]" -and $_.Extension -in ".sqlite", ".sqlite3", ".db")
    }
    if ($forbidden) {
        throw "Forbidden files were found in the private bundle staging directory."
    }

    $bundleManifest = [ordered]@{
        schema_version = "private-web-bundle-v1"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        retrieval_index_id = $manifest.index_id
        chunk_count = $manifest.chunk_count
        includes_environment = $false
        includes_pdfs = $false
    }
    $bundleManifest | ConvertTo-Json | Set-Content -Encoding UTF8 "$stageRoot\bundle-manifest.json"

    & tar.exe -czf $archivePath -C $stageParent $bundleName
    if ($LASTEXITCODE -ne 0) { throw "tar.exe failed with exit code $LASTEXITCODE" }
    Write-Output $archivePath
}
finally {
    if (Test-Path -LiteralPath $stageParent) {
        $resolvedStage = [IO.Path]::GetFullPath($stageParent)
        if ($resolvedStage.StartsWith($outputParent, [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolvedStage -Recurse -Force
        }
    }
}
