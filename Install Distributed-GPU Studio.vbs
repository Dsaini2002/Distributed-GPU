Option Explicit
Dim fso, shell, script, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
script = fso.BuildPath(fso.GetParentFolderName(WScript.ScriptFullName), "Desktop-App\Install-Desktop.ps1")
If Not fso.FileExists(script) Then
  MsgBox "App files are missing. Extract the entire ZIP first.", vbCritical, "Distributed-GPU Studio"
  WScript.Quit 1
End If
cmd = "powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -WindowStyle Hidden -File " & Chr(34) & script & Chr(34)
shell.Run cmd, 0, False
