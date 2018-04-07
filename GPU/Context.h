#pragma once

#include "gpu_global.h"
#ifdef GPU_LIB
#include <helper_cuda.h>
#endif
namespace TRN
{
	namespace GPU
	{
		class Context
		{
		public :
			static const std::size_t STREAM_NUMBER;
			static const std::size_t EVENT_NUMBER;
		private: 

		

			static const std::size_t DEFAULT_DIV;
			static const std::size_t DEFAULT_DIMS;
			static const std::size_t DEFAULT_DYNAMIC_MEMORY_SIZE;

			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			GPU_EXPORT Context(const int &device);
			GPU_EXPORT ~Context();

#ifdef GPU_LIB
		public :
			void dispose();
			void toggle();
			const std::size_t &get_stride_alignment();
			const int &get_device();
			const cudaStream_t *get_streams();
			const cublasHandle_t *get_handles();
			const cudaEvent_t *get_events();
			//const curandGenerator_t &get_generator();
			const std::string &get_name();
			template < class T >
			static void get_block_dims(T func, const std::vector < std::size_t> &dimensions, dim3 &grid, dim3 &block, const std::size_t &div = DEFAULT_DIV, const std::size_t &dynamic_memory_size = DEFAULT_DYNAMIC_MEMORY_SIZE)
			{
//				std::size_t dim_x, dim_y, dim_z;
				int minGridSize, maxBlockSize;

				checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, func, dynamic_memory_size));

				/*int device;
				cudaDeviceProp prop;

				checkCudaErrors(cudaGetDevice(&device));
				checkCudaErrors(cudaGetDeviceProperties(&prop, device));*/

//				const std::size_t WARP_PER_BLOCK = 2;
				//const std::size_t THREAD_PER_WARP = 32;
//				const std::size_t THREADS_PER_BLOCK = WARP_PER_BLOCK * THREAD_PER_WARP;
				maxBlockSize /= div;
	
				//auto number_of_blocks = maxBlockSize / THREADS_PER_BLOCK;
				switch (dimensions.size())
				{
				case 1:
					block.x = maxBlockSize;
					block.y = 1;
					block.z = 1;
					grid.x = (dimensions[0] + block.x - 1) / block.x;
					grid.y = 1;
					grid.z = 1;
					
					break;
				case 2:
					block.x = maxBlockSize / 4;
					grid.x = (dimensions[0] + block.x - 1) / block.x;

					block.y = std::min((std::size_t)4UL, dimensions[1]);
					grid.y = (dimensions[1] + block.y - 1) / block.y;

					block.z = 1;
					grid.z = 1;

					break;
				case 3:
					block.x = maxBlockSize / 8;
					grid.x = (dimensions[0] + block.x - 1) / block.x;

					block.y = std::min((std::size_t)4UL, dimensions[1]);
					grid.y = (dimensions[1] + block.y - 1) / block.y;

					block.z = std::min((std::size_t)2UL, dimensions[2]);
					grid.z = (dimensions[2] + block.z - 1) / block.z;

					break;

				default:
					throw std::invalid_argument("Unexpected number of dimensions " + dimensions.size());
				}
				/*block.x = std::min(block.x, (unsigned int)prop.maxThreadsDim[0]);
				block.y = std::min(block.y, (unsigned int)prop.maxThreadsDim[1]);
				block.z = std::min(block.z, (unsigned int)prop.maxThreadsDim[2]);

				grid.x = std::min(grid.x, (unsigned int)prop.maxGridSize[0]);
				grid.y = std::min(grid.y, (unsigned int)prop.maxGridSize[1]);
				grid.z = std::min(grid.z, (unsigned int)prop.maxGridSize[2]);*/


				/*auto gridSize = grid.x * grid.y * grid.z;
				INFORMATION "gridSize = " << gridSize << ", minGridSize = " << minGridSize ;*/
			}

			template < class T, typename UnaryFunction >
			static void get_block_dims_variable_shared_memory(T func, const std::vector<std::size_t> &dimensions, dim3 &grid, dim3 &block, UnaryFunction blockSizeToDynamicSMemSize)
			{
		
				int minGridSize, maxBlockSize;
				checkCudaErrors(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &maxBlockSize, func, blockSizeToDynamicSMemSize));

				switch (dimensions.size())
				{
					case 1:
						block.x = maxBlockSize;
						block.y = 1;
						block.z = 1;
						grid.x = (dimensions[0] + block.x - 1) / block.x;
						grid.y = 1;
						grid.z = 1;

						break;
					case 2:
						block.x = maxBlockSize / 32;
						block.y = 32;
						block.z = 1;
						grid.x = (dimensions[0] + block.x - 1) / block.x;
						grid.y = (dimensions[1] + block.y - 1) / block.y;
						grid.z = 1;
						break;
					case 3:
						block.x = maxBlockSize / 32;
						block.y = 32;
						block.z = 1;
						grid.x = (dimensions[0] + block.x - 1) / block.x;
						grid.y = (dimensions[1] + block.y - 1) / block.y;
						grid.z = (dimensions[2] + block.z - 1) / block.z;

						break;

					default:
						throw std::invalid_argument("Unexpected number of dimensions " + dimensions.size());
				}

			
			}
#endif

		public :
			static std::shared_ptr<Context> GPU_EXPORT create(const int &device);
		};

	};
};