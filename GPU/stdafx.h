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
#include <algorithm>
#include <random>
#include <iomanip> // setprecision
#include <sstream>
// CUDA runtime


#include <cuda.h>
#include <cublas.h>
#include <cublas_v2.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

//#include <npp.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>



