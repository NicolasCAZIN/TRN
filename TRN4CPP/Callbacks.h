#pragma once

#include "trn4cpp_global.h"
#include "Basic.h"

namespace TRN4CPP
{
	namespace Plugin
	{
		namespace Callbacks
		{
			class TRN4CPP_EXPORT Interface : public Plugin::Basic::Interface
			{
			public:
				virtual	void callback_measurement_readout_raw(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) = 0;
				virtual void callback_measurement_position_raw(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) = 0;
				virtual void callback_measurement_readout_mean_square_error(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) = 0;
				virtual void callback_measurement_readout_frechet_distance(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) = 0;
				virtual void callback_measurement_position_mean_square_error(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) = 0;
				virtual void callback_measurement_position_frechet_distance(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) = 0;

				virtual void callback_performances(const unsigned int &id, const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds) = 0;
				virtual void callback_states(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) = 0;
				virtual void callback_weights(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) = 0;
				virtual void callback_scheduling(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations) = 0;
			};

			void TRN4CPP_EXPORT		initialize(const std::string &library_path, const std::string &name, const std::map<std::string, std::string> &arguments);
		};
	}

	namespace Simulation
	{
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
				namespace Raw
				{
					void TRN4CPP_EXPORT 	install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);
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
				namespace Raw
				{
					void TRN4CPP_EXPORT 	install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);
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
