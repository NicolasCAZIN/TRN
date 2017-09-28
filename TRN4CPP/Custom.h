#pragma once

#include "trn4cpp_global.h"

namespace TRN4CPP
{
	namespace Simulation
	{
		namespace Scheduler
		{
			namespace Custom
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
					std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply);
			};
			namespace Mutator
			{
				namespace Custom
				{
					void TRN4CPP_EXPORT  install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
						std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply);
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
						void TRN4CPP_EXPORT		install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
				namespace Feedback
				{
					namespace Custom
					{
						void TRN4CPP_EXPORT		install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
				namespace Recurrent
				{
					namespace Custom
					{
						void TRN4CPP_EXPORT		install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
				namespace Readout
				{
					namespace Custom
					{
						void TRN4CPP_EXPORT		install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
			};
		};
		namespace Measurement
		{
			namespace Readout
			{
				namespace Custom
				{
					void TRN4CPP_EXPORT 	install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);
				};
			};
			namespace Position
			{
				namespace Custom
				{
					void TRN4CPP_EXPORT 	install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);
				};
			};
		};

	};
};
