param([Parameter(Mandatory=$true)] $Version)
cd .github\cuda-$Version
choco pack
choco install --verbose -s . cuda
if ( $? -eq $false )
{
	Get-Content -Tail 100 C:\ProgramData\chocolatey\logs\chocolatey.log 
	Exit 1
}
