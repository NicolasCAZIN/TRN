#pragma once

#include "trn4cpp_global.h"

namespace TRN4CPP
{
	namespace Simulation
	{
		namespace Loop
		{
			namespace Readout
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
					std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply);
			};
			namespace Position
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
					std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply);
			};
		};

		namespace Measurement
		{
			namespace Readout
			{
				namespace MeanSquareError
				{
					void TRN4CPP_EXPORT 	install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);

				};
				namespace FrechetDistance
				{
					void TRN4CPP_EXPORT 	install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);

				};
			};
			namespace Position
			{
				namespace MeanSquareError
				{
					void TRN4CPP_EXPORT 	install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);

				};
				namespace FrechetDistance
				{
					void TRN4CPP_EXPORT 	install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);

				};
			};
		};
		namespace Recording
		{
			namespace States
			{
				void TRN4CPP_EXPORT  	install(const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &request);
			};
			namespace Weights
			{
				void TRN4CPP_EXPORT  	install(const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &request);
			};
			namespace Performances
			{
				void TRN4CPP_EXPORT  	install(const std::function<void(const unsigned int &id, const std::string &phase, const std::size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &request);
			};
			namespace Scheduling
			{
				void TRN4CPP_EXPORT  	install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &request);
			};
		};
	};
};
