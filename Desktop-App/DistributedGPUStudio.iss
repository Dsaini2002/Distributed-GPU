#define MyAppName "Distributed-GPU Studio"
#define MyAppVersion "0.1.0-beta"
#define MyAppPublisher "Distributed-GPU Studio"
#define MyAppExeName "DistributedGPUStudio.exe"
[Setup]
AppId={{A51D8713-AB7C-44F1-AD48-6055D22A1681}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\DistributedGPUStudio
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer-output
OutputBaseFilename=DistributedGPU-Setup-Beta
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
[Files]
Source: "dist\DistributedGPUStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"
[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Distributed-GPU Studio"; Flags: nowait postinstall skipifsilent
