; Inno Setup 6 — builds a Windows installer from the PyInstaller onedir output.
; Prereq: run packaging\scripts\build_pyinstaller.ps1 then this script with ISCC.exe in PATH.
;
; Optional branding: place wizard-large.bmp / wizard-small.bmp in packaging\assets\installer
; (run packaging\scripts\prepare_inno_wizard_images.ps1 to generate BMPs from your PNGs).

#define MyAppName "BetaSafe"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "missbijoux"
#define MyAppURL "https://github.com/missbijoux/Beta-Safe"
#define MyAppExeName "BetaSafe.exe"
#define PyInstallerOut "..\..\dist\BetaSafe"

[Setup]
AppId={{A8C9E2F1-4B0D-4C7A-9E3B-0D1F2E3A4B5C}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\out\installer
OutputBaseFilename=BetaSafe-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
CloseApplications=no
ArchitecturesInstallIn64BitMode=x64
SetupIconFile=..\icons\app.ico

#if FileExists(AddBackslash(SourcePath) + "..\assets\installer\wizard-large.bmp")
WizardImageFile=..\assets\installer\wizard-large.bmp
#endif
#if FileExists(AddBackslash(SourcePath) + "..\assets\installer\wizard-small.bmp")
WizardSmallImageFile=..\assets\installer\wizard-small.bmp
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#PyInstallerOut}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
