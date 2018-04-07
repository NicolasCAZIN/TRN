#pragma once

#include "trn4cpp_global.h"
#include "Basic.h"

namespace TRN4CPP
{
	namespace Plugin
	{
		namespace Custom
		{
			class TRN4CPP_EXPORT Interface : public Plugin::Basic::Interface
			{
			public:
				virtual void callback_position(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols) = 0;
				virtual void install_position(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &functor) = 0;

				virtual void callback_stimulus(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols) = 0;
				virtual void install_stimulus(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &functor) = 0;

				virtual void callback_mutator(const unsigned long long &simulation_id,  const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations) = 0;
				virtual void install_mutator(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor) = 0;

				virtual void callback_scheduler(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations) = 0;
				virtual	void install_scheduler(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor) = 0;

				virtual void callback_feedforward(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) = 0;
				virtual void install_feedforward(const std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &functor) = 0;

				virtual void callback_readout(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) = 0;
				virtual void install_readout(const std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &functor) = 0;

				virtual void callback_recurrent(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols) = 0;
				virtual void install_recurrent(const std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &functor) = 0;
			};

			void TRN4CPP_EXPORT		initialize(const std::string &library_path, const std::string &name, const std::map<std::string, std::string> &arguments);
			void uninitialize();
		};
	}

	namespace Simulation
	{
		namespace Loop
		{
			namespace Custom
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
					std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply);
			};
		
		};

		namespace Scheduler
		{
			namespace Custom
			{
				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
					std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply);
			};
			namespace Mutator
			{
				namespace Custom
				{
					void TRN4CPP_EXPORT  install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
						std::function<void(const unsigned long long &simulation_id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply);
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
						void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
				namespace Feedback
				{
					namespace Custom
					{
						void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
				namespace Recurrent
				{
					namespace Custom
					{
						void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
				namespace Readout
				{
					namespace Custom
					{
						void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
							std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
					};
				};
			};
		};

		namespace Encoder
		{
			namespace Custom // position
			{

				void TRN4CPP_EXPORT		install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
						std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply);
			
			};
		};
	};
};
