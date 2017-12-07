#pragma once

#include "Basic.h"

namespace TRN4CPP
{
	namespace Engine
	{
		namespace Execution
		{
			void TRN4CPP_EXPORT  	run();
		};
	};

	namespace Simulation
	{
		void TRN4CPP_EXPORT  	allocate(const unsigned long long &id);
		void TRN4CPP_EXPORT  	deallocate(const unsigned long long &id);

		void TRN4CPP_EXPORT  	train(const unsigned long long &id, const std::string &label, const std::string &incoming, const std::string &expected);
		void TRN4CPP_EXPORT  	test(const unsigned long long &id, const std::string &sequence, const std::string &incoming, const std::string &expected, const unsigned int &preamble, const bool &autonomous, const unsigned int &supplementary_generations = 0);

		void TRN4CPP_EXPORT  	declare_sequence(const unsigned long long &id, const std::string &label, const std::string &tag, const std::vector<float> &sequence, const std::size_t &observations);
		void TRN4CPP_EXPORT  	declare_set(const unsigned long long &id, const std::string &label, const std::string &tag, const std::vector<std::string> &labels);
		void TRN4CPP_EXPORT 	configure_begin(const unsigned long long &id);
		void TRN4CPP_EXPORT 	configure_end(const unsigned long long &id);

		namespace Loop
		{
			namespace Copy
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size);
			};
			namespace Custom
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size);
			};
			namespace SpatialFilter
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
												  const std::size_t &rows, const std::size_t &cols, const std::pair<float, float> &x, const std::pair<float, float> &y,
												  const std::vector<float> response, const float &sigma, const float &radius, const float &scale, const std::string &tag);
			};
		};
		namespace Scheduler
		{
			namespace Tiled
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const unsigned int &epochs);
			};
			namespace Snippets
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag = DEFAULT_TAG);
			};
			namespace Custom
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const unsigned long &seed, const std::string &tag);
			};

			namespace Mutator
			{
				namespace Shuffle
				{
					void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const unsigned long &seed);
				};
				namespace Reverse
				{
					void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const unsigned long &seed, const float &rate, const std::size_t &size);
				};
				namespace Punch
				{
					void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &counter);
				};
				namespace Custom
				{
					void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const unsigned long &seed);
				};

			};
		};
		namespace Reservoir
		{
			namespace WidrowHoff
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate, const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size);
			};

			namespace Weights
			{
				namespace Feedforward
				{
					namespace Gaussian
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const float &mu, const float &sigma);
					};

					namespace Uniform
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const float &a, const float &b, const float &sparsity);
					};

					namespace Custom
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id);
					};
				};
				namespace Feedback
				{
					namespace Gaussian
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const float &mu, const float &sigma);
					};

					namespace Uniform
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const float &a, const float &b, const float &sparsity);
					};

					namespace Custom
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id);
					};
				};
				namespace Recurrent
				{
					namespace Gaussian
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const float &mu, const float &sigma);
					};

					namespace Uniform
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const float &a, const float &b, const float &sparsity);
					};

					namespace Custom
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id);
					};
				};
				namespace Readout
				{
					namespace Gaussian
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const float &mu, const float &sigma);
					};

					namespace Uniform
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const float &a, const float &b, const float &sparsity);
					};

					namespace Custom
					{
						void TRN4CPP_EXPORT  	configure(const unsigned long long &id);
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
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size);

				};
				namespace FrechetDistance
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size);

				};
				namespace Custom
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size);
				};
			};
			namespace Position
			{
				namespace MeanSquareError
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size);

				};
				namespace FrechetDistance
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size);

				};
				namespace Custom
				{
					void TRN4CPP_EXPORT 	configure(const unsigned long long &id, const std::size_t &batch_size);

				};
			};

		};
		namespace Recording
		{
			namespace States
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const bool &train, const bool &prime, const bool &generate);
			};
			namespace Weights
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const bool &initialize, const bool &train);
			};
			namespace Performances
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id, const bool &train, const bool &prime, const bool &generate);
			};
			namespace Scheduling
			{
				void TRN4CPP_EXPORT  	configure(const unsigned long long &id);
			};
		};
	};
};
