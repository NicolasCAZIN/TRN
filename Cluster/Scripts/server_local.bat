@ECHO OFF
SETLOCAL enableextensions
setlocal EnableDelayedExpansion

SET HOST=%1
SET PORT=%2

SET INDEX=%3 %4 %5 %6

IF "%HOST%" == "" (
ECHO A server host must be specified
EXIT
)
IF "%PORT%" == "" (
ECHO A server port must be specified
EXIT
)
IF "%INDEX%" == "" (
ECHO A device index must be specified
EXIT
)


Server --host %HOST% --port %PORT% --backend local --index %INDEX% --logging information

