"P:\Programs\WinSCP\WinSCP.com" ^
  /ini=nul ^
  /command ^
    "open sftp://mrsftp:Cetus@216.165.115.136/ -hostkey=""ssh-ed25519 256 TvJDI2rNEP81saVn0wuoT8qIBSYJfkpawYEcEQbigVk=""" ^
    "lcd ""S:\Computer Science\AI\Policy Envelopes""" ^
    "cd /uploads" ^
    "put PolicyEnvelope" ^
    "exit"

set WINSCP_RESULT=%ERRORLEVEL%
if %WINSCP_RESULT% equ 0 (
  echo Success
) else (
  echo Error
)

exit /b %WINSCP_RESULT%