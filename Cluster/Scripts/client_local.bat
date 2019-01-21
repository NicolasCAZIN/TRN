@ECHO OFF
SETLOCAL enableextensions
setlocal EnableDelayedExpansion
SET SCENARIO=%1
SET INDEX=%2 %3 %4 %5 %6
IF "%SCENARIO%" == "" (
ECHO A scenario file must be specified
EXIT
)
IF "%INDEX%" == "" (
ECHO A device index must be specified
EXIT
)
Client --filename %SCENARIO% --backend local --index %INDEX% --logging information
