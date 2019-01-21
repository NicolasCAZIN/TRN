@ECHO OFF
SETLOCAL enableextensions
setlocal EnableDelayedExpansion

SET SUBSCRIBERS=%1
SET HOST=%2
SET PORT=%3

IF "%TRN_SIMULATOR%" == "" (
ECHO Variable TRN_SIMULATOR is not set
EXIT
)

IF "%SUBSCRIBERS%" == "" (
ECHO A subscriber number must be specified
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



ECHO -host %HOST% -n 1 %TRN_SIMULATOR%\Server.exe --host 0.0.0.0 --port %PORT% --backend distributed --logging information > backend.mpi

set argCount=0
for %%x in (%*) do (
   set /A argCount+=1
   set "argVec[!argCount!]=%%~x"
)

set count=0
for /L %%k in (4,2,%argCount%) do (
	 set /A count+=1
	 set "workers[!count!]=!argVec[%%k]!"
)
set count=0
for /L %%k in (5,2,%argCount%) do (
	 set /A count+=1
	 set "indexes[!count!]=!argVec[%%k]!"
)
for /L %%k in (1,1,%count%) do (
	ECHO -host !workers[%%k]! -n %SUBSCRIBERS% %TRN_SIMULATOR%\Worker.exe --index !indexes[%%k]! >> backend.mpi
)


