@ECHO OFF
SETLOCAL enableextensions
setlocal EnableDelayedExpansion
SET SCENARIO=%1
SET HOST=%2
SET PORT=%3
IF "%SCENARIO%" == "" (
ECHO A Scenario file must be specified
EXIT
)
IF "%HOST%" == "" (
ECHO A server host must be specified
EXIT
)
IF "%PORT%" == "" (
ECHO A server port must be specified
EXIT
)
Client --filename %SCENARIO% --backend remote --host %HOST% --port %PORT% --logging information