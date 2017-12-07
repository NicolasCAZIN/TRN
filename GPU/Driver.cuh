#pragma once
#include <string>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <helper_cuda.h>
#include <sstream>
#include "Helper/Logger.h"

/*template< typename T >
void throw_on_cuda_error(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		ERROR_LOGGER << "Cuda call failed with error code " << _cudaGetErrorEnum(result) << " from function " << func << " line " << line << " file " << file;
		const int code = static_cast<unsigned int>(result);
		std::stringstream ss;
		ss << file << " (" << line << ") in function " << " \"" << func << " \""<<  _cudaGetErrorEnum(result) ;
		std::string file_and_line_func;
		ss >> file_and_line_func;

		throw std::runtime_error(file_and_line_func);
	}
}
#ifdef checkCudaErrors
#undef checkCudaErrors
#endif 
#define checkCudaErrors(val)          throw_on_cuda_error ( (val), #val, __FILE__, __LINE__ )*/


#define gpuErrchk(ans) { gpuAssert((ans),#ans, __FILE__, __LINE__); }
template< typename T >
__device__
inline void gpuAssert(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		printf("GPUassert: code %d function \"%s\" file %s line %d\n", result, func, file, line);
		
	}
}