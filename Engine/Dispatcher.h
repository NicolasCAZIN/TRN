#pragma once

#include "engine_global.h"
#include "Broker.h"
#include "Helper/Observer.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Dispatcher :
			public TRN::Engine::Broker,
			public TRN::Helper::Observable<	TRN::Engine::Message<TRN::Engine::EXIT>>
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Dispatcher(const std::shared_ptr<TRN::Engine::Communicator> &to_workers);
			virtual ~Dispatcher();


		public :
			void register_frontend(const unsigned short &frontend, const std::shared_ptr<TRN::Engine::Communicator> &communicator);
			void unregister_frontend(const unsigned short &frontend);

	

		protected :
			virtual void callback_completed() override;
			virtual void callback_configured(const unsigned long long &simulation_id) override;
			virtual void callback_ack(const unsigned long long &simulation_id, const std::size_t &counter, const bool &success, const std::string &cause) override;
			virtual void callback_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name) override;
			virtual void callback_allocated(const unsigned long long &simulation_id, const int &rank) override;
			virtual void callback_deallocated(const unsigned long long &simulation_id, const int &rank) override;
			virtual void callback_exit(const unsigned short &number, const int &rank) override;
			virtual void callback_terminated(const int &rank) override;
			virtual void callback_trained(const unsigned long long &simulation_id, const unsigned long long &evaluation_id) override;
			virtual void callback_primed(const unsigned long long &simulation_id, const unsigned long long &evaluation_id) override;
			virtual void callback_tested(const unsigned long long &simulation_id, const unsigned long long &evaluation_id) override;
			virtual void callback_error(const std::string &message) override;
			virtual void callback_information(const std::string &message) override;
			virtual void callback_warning(const std::string &message) override;


			virtual void callback_measurement_readout_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_readout_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_readout_custom(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_position_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_position_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_position_custom(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override;

			virtual void callback_performances(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second) override;
			virtual void callback_states(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override;
			virtual void callback_weights(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override;
			virtual void callback_position(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols) override;
			virtual void callback_stimulus(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols) override;

			virtual void callback_mutator(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations) override;
			virtual void callback_scheduler(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed,  const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations) override;
			virtual void callback_scheduling(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations) override;

			virtual void callback_feedforward(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_readout(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_recurrent(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols) override;


		public :
			static std::shared_ptr<TRN::Engine::Dispatcher> create(const std::shared_ptr<TRN::Engine::Communicator> &to_workers);
		};
	}
};