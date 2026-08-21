param(
    [ValidateRange(1, 3650)]
    [int]$OlderThanDays = 14,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeRoot = Join-Path $repositoryRoot "backend\runtime"
$jobsRoot = Join-Path $runtimeRoot "jobs"

if (-not (Test-Path -LiteralPath $jobsRoot -PathType Container)) {
    Write-Host "Không có thư mục runtime jobs để dọn: $jobsRoot"
    exit 0
}

$resolvedRuntime = (Resolve-Path -LiteralPath $runtimeRoot).Path
$resolvedJobs = (Resolve-Path -LiteralPath $jobsRoot).Path
if (-not $resolvedJobs.StartsWith($resolvedRuntime, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Từ chối dọn vì đường dẫn jobs nằm ngoài backend/runtime."
}

$cutoff = (Get-Date).AddDays(-$OlderThanDays)
$targets = @(
    Get-ChildItem -LiteralPath $resolvedJobs -Directory -Force |
        Where-Object { $_.LastWriteTime -lt $cutoff }
)

$totalBytes = 0L
foreach ($target in $targets) {
    $files = Get-ChildItem -LiteralPath $target.FullName -Recurse -File -Force -ErrorAction SilentlyContinue
    $totalBytes += [long](($files | Measure-Object -Property Length -Sum).Sum)
}

Write-Host ("Tìm thấy {0} job cũ hơn {1} ngày, tổng {2:N2} MB." -f $targets.Count, $OlderThanDays, ($totalBytes / 1MB))
if (-not $Apply) {
    Write-Host "Đây là chế độ xem trước. Thêm -Apply để xóa các job trên."
    $targets | Select-Object Name, LastWriteTime | Format-Table -AutoSize
    exit 0
}

foreach ($target in $targets) {
    $resolvedTarget = (Resolve-Path -LiteralPath $target.FullName).Path
    if ((Split-Path -Parent $resolvedTarget) -ne $resolvedJobs) {
        throw "Từ chối xóa đường dẫn ngoài jobs root: $resolvedTarget"
    }
    Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
}

Write-Host ("Đã xóa {0} job runtime cũ; dữ liệu này không thể khôi phục từ script." -f $targets.Count)
