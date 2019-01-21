@ECHO OFF
SETLOCAL enableextensions
setlocal EnableDelayedExpansion

SET TRN_ROOT=C:\Users\cazin\OneDrive\Documents\Visual Studio 2015\Projects\TRN\x64\Release
REM SET TRN_ROOT=%1

IF "%BOOST_ROOT%" == "" (
ECHO Variable BOOST_ROOT is not set
EXIT
)
IF "%CUDA_PATH%" == "" (
ECHO Variable CUDA_PATH is not set
EXIT
)
IF "%MKL_ROOT%" == "" (
ECHO Variable MKL_ROOT is not set
EXIT
)
IF "%OPENCV_ROOT%" == "" (
ECHO Variable OPENCV_ROOT is not set
EXIT
)
IF "%MATLAB_ROOT%" == "" (
ECHO Variable MATLAB_ROOT is not set
EXIT
)
IF "%TRN_TEMP%" == "" (
ECHO Variable TRN_TEMP is not set
EXIT
)
IF NOT EXIST "%TRN_TEMP%" (
	ECHO Directory "%TRN_TEMP%" does not exist. Creating directory.
	MKDIR "%TRN_TEMP%"
	IF  %ERRORLEVEL% NEQ 0 (
		ECHO failed with return code  %ERRORLEVEL% 
		EXIT
	)
)

REM DEL /S /F /Q "%TRN_TEMP%"\*

REM TRN Files
ECHO Copying TRN Runtime files
COPY "%TRN_ROOT%"\*.dll "%TRN_TEMP%"
COPY "%TRN_ROOT%"\*.exe "%TRN_TEMP%"
REM COPY "%TRN_ROOT%"\*.pdb "%TRN_TEMP%"
COPY "%TRN_ROOT%"\*.mexw64 "%TRN_TEMP%"
COPY "%TRN_ROOT%"\..\..\TRN4JAVA\*.jar "%TRN_TEMP%"
COPY "%TRN_ROOT%"\..\..\TRN4JAVA\*.java "%TRN_TEMP%"
COPY "%TRN_ROOT%"\..\..\TRN4JAVA\test.bat "%TRN_TEMP%"
COPY "%TRN_ROOT%"\..\..\Cluster\Scripts\*.bat "%TRN_TEMP%"
COPY "%TRN_ROOT%"\..\..\Cluster\Configuration\*.mpi "%TRN_TEMP%"

REM COPY "%TRN_ROOT%"\*.pdb "%TRN_TEMP%"

REM VS Files
ECHO Copying Visual Studio Runtime files
COPY "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x64\Microsoft.VC140.CRT\*.dll" "%TRN_TEMP%"
COPY "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x64\Microsoft.VC140.OpenMP\*.dll" "%TRN_TEMP%"

REM MATLAB Files
ECHO Copying MATLAB Runtime files
COPY "%MATLAB_ROOT%"\bin\win64\libmat.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\libmx.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\libmwfl.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\libmwi18n.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\libmwfoundation_usm.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\libmwresource_core.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\boost_filesystem-vc140-mt-1_56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\boost_system-vc140-mt-1_56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\boost_thread-vc140-mt-1_56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\boost_regex-vc140-mt-1_56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\boost_chrono-vc140-mt-1_56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\boost_serialization-vc140-mt-1_56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\boost_signals-vc140-mt-1_56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\boost_log-vc140-mt-1_56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\boost_date_time-vc140-mt-1_56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\libexpat.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\libut.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\icuuc56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\icuin56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\icuio56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\icudt56.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\tbb.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\tbbmalloc.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\msvcp120.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\msvcr120.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\vcomp120.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\hdf5.dll "%TRN_TEMP%"
COPY "%MATLAB_ROOT%"\bin\win64\zlib1.dll "%TRN_TEMP%"
REM OPENCV Files
ECHO Copying OPENCV Runtime files
COPY "%OPENCV_ROOT%"\x64\vc14\bin\opencv_world320.dll "%TRN_TEMP%"

REM BOOST Files
ECHO Copying BOOST Runtime files
COPY "%BOOST_ROOT%"\lib\boost_chrono-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_thread-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_date_time-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_regex-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_log-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_log_setup-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_bzip2-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_zlib-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_iostreams-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_serialization-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_system-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_filesystem-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_mpi-vc140-mt-1_62.dll "%TRN_TEMP%"
COPY "%BOOST_ROOT%"\lib\boost_program_options-vc140-mt-1_62.dll "%TRN_TEMP%"


REM MKL Files
ECHO Copying MKL Runtime files
COPY "%MKL_ROOT%"\..\redist\intel64_win\mkl\*.dll "%TRN_TEMP%"
REM COPY "%MKL_ROOT%"\..\redist\intel64_win\mkl\mkl_def.dll "%TRN_TEMP%"
REM COPY "%MKL_ROOT%"\..\redist\intel64_win\mkl\mkl_core.dll "%TRN_TEMP%"
REM COPY "%MKL_ROOT%"\..\redist\intel64_win\mkl\mkl_avx.dll "%TRN_TEMP%"
REM COPY "%MKL_ROOT%"\..\redist\intel64_win\mkl\mkl_avx2.dll "%TRN_TEMP%"
REM COPY "%MKL_ROOT%"\..\redist\intel64_win\mkl\mkl_intel_thread.dll "%TRN_TEMP%"
COPY "%MKL_ROOT%"\..\redist\intel64_win\compiler\libiomp5md.dll "%TRN_TEMP%"

REM CUDA Files
ECHO Copying CUDA Runtime files
COPY "%CUDA_PATH%"\bin\cudart64_100.dll "%TRN_TEMP%"
COPY "%CUDA_PATH%"\bin\curand64_100.dll "%TRN_TEMP%"
COPY "%CUDA_PATH%"\bin\cublas64_100.dll "%TRN_TEMP%"
REM OpenCV Files
REM MATLAB Files

 REM  COPY "%TRN_TEMP%" \\KAMINO\Simulator
REM FOR %%H IN (DEATHSTAR) DO (
REM  FOR %%H IN (ALDERAAN,  DAGOBAH, DEATHSTAR, HOTH,KAMINO, KASHYYYK, NABOO, TATOOINE, BESPIN) DO (
 FOR %%H IN (DEATHSTAR, HOTH) DO (
  ECHO Deploying simulator to %%H
   DEL /S /F /Q \\%%H\Simulator\*
  START ROBOCOPY %TRN_TEMP% "\\%%H\Simulator"
  )
