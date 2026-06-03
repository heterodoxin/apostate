$host.UI.RawUI.WindowTitle = 'apostate'
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"
& python -m apostate @args
