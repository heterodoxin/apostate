# resize
$maxWidth = 240
$maxHeight = 60

# console
$console = [System.Console]
$host.UI.RawUI.BufferSize = New-Object System.Management.Automation.Host.Size($maxWidth, $maxHeight)
$host.UI.RawUI.WindowSize = New-Object System.Management.Automation.Host.Size([Math]::Min($maxWidth, $host.UI.RawUI.LargestWindowWidth), [Math]::Min($maxHeight, $host.UI.RawUI.LargestWindowHeight))
$host.UI.RawUI.WindowTitle = 'apostate'

# launch
& node "$PSScriptRoot/main.js" @args
