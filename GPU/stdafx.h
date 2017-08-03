#pragma once
#ifdef USE_VLD
#include <vld.h>
#endif 
#include <memory>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <mutex>
#include <iostream>
#include <cassert>
#include <random>
// CUDA runtime
#include <opencv2/core.hpp>

#include <cuda.h>
#include <cublas_v2.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <helper_cuda.h>
//#include <npp.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>



