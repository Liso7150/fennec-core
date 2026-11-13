param(
  [string]$Root = (Join-Path $PSScriptRoot 'target')
)
# Sign freshly-built Rust binaries with the local dev code-signing cert so Smart
# App Control on THIS machine will execute them (new unsigned hashes are blocked,
# os error 4551). Local-only trust: the cert lives in CurrentUser\Root and
# TrustedPublisher; this does NOT make the app run on other users' SAC machines
# (that needs a trusted-CA signature, see task #105). Reversible: remove the cert.
$ErrorActionPreference = 'Stop'
$cert = Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Subject -eq 'CN=Fennec Local Dev Signing' } | Select-Object -First 1
if (-not $cert) { throw 'Fennec Local Dev Signing cert missing; recreate it (CodeSigningCert) and trust it in CurrentUser\Root + TrustedPublisher.' }
$signed = 0
Get-ChildItem -Path $Root -Recurse -Include *.exe, *.dll -ErrorAction SilentlyContinue | ForEach-Object {
  $sig = Get-AuthenticodeSignature $_.FullName
  if ($sig.Status -ne 'Valid') {
    Set-AuthenticodeSignature -FilePath $_.FullName -Certificate $cert -HashAlgorithm SHA256 | Out-Null
    $signed++
  }
}
Write-Output "signed $signed binaries under $Root"