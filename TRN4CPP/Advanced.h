#pragma once

#include "Basic.h"

namespace TRN4CPP
{
	namespace Engine
	{
		namespace Events
		{
			namespace Configured
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &id)> &functor);
			};
			namespace Trained
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &id)> &functor);
			};
			namespace Primed
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &id)> &functor);
			};
			namespace Tested
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &id)> &functor);
			};
			namespace Completed
			{
				void TRN4CPP_EXPORT		install(const std::function<void()> &functor);
			};
			namespace Ack
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &id, const std::size_t &counter, const bool &success, const std::string &cause)> &functor);
			};
			namespace Processor
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &functor);
			};
			namespace Allocated
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &id, const int &rank)> &functor);
			};
			namespace Deallocated
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &id, const int &rank)> &functor);
			};
		};

	}
	namespace Simulation
	{
		namespace Loop
		{
			namespace Custom
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size,
					const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
					std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply
				);
		
			};
			namespace SpatialFilter
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
					const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
					std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
					const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
					std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
					const std::size_t &rows, const std::size_t &cols, const std::pair<float, float> &x, const std::pair<float, float> &y,
					const std::vector<float> response, const float &sigma, const float &radius, const float &scale, const std::string &tag);
			};
		};
		namespace Scheduler
		{
			namespace Custom
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const unsigned long &seed,
					const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
					std::function<void(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply, const std::string &tag);
			};

			namespace Mutator
			{
				namespace Custom
				{
					void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const unsigned long &seed,
						const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
						std::function<void(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply
					);
				};
			};
		};
		namespace Reservoir
		{
			namespace Weights
			{
				namespace Feedforward
				{
					namespace Custom
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id,
							const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
				namespace Feedback
				{
					namespace Custom
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id,
							const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
				namespace Recurrent
				{
					namespace Custom
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id,
							const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
				namespace Readout
				{
					namespace Custom
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id,
							const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
			};
		};
		namespace Measurement
		{
			namespace Readout
			{
				namespace MeanSquareError
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
				};
				namespace FrechetDistance
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
				};
				namespace Custom
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);
				};
			};
			namespace Position
			{
				namespace MeanSquareError
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
				};
				namespace FrechetDistance
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
				};
				namespace Custom
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);
				};
			};

		};
		namespace Recording
		{
			namespace States
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &train, const bool &prime, const bool &generate);
			};
			namespace Weights
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initialize, const bool &train);
			};
			namespace Performances
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> &functor, const bool &train, const bool &prime, const bool &generate);
			};
			namespace Scheduling
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor);
			};
		};
	};
};