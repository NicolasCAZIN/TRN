#pragma once

#include "engine_global.h"
#include "Broker.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Frontend : public TRN::Engine::Broker
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Frontend(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const std::shared_ptr<TRN::Engine::Executor> &to_caller);
			virtual ~Frontend();

		

		public:
			void install_completed(const std::function<void()> &functor);
			void install_ack(const std::function<void(const unsigned int &id, const std::size_t &number, const bool &success, const std::string &cause)> &functor);
			void install_processor(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &functor);
			void install_allocated(const std::function<void(const unsigned int &id, const int &rank)> &functor);
			void install_deallocated(const std::function<void(const unsigned int &id, const int &rank)> &functor);
			void install_quit(const std::function<void(const int &rank)> &functor);
			void install_trained(const std::function<void(const unsigned int &id)> &functor);
			void install_configured(const std::function<void(const unsigned int &id)> &functor);
			void install_primed(const std::function<void(const unsigned int &id)> &functor);
			void install_tested(const std::function<void(const unsigned int &id)> &functor);
			void install_error(const std::function<void(const std::string &message)> &functor);
			void install_information(const std::function<void(const std::string &message)> &functor);
			void install_warning(const std::function<void(const std::string &message)> &functor);


			void install_measurement_readout_mean_square_error(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor);
			void install_measurement_readout_frechet_distance(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor);
			void install_measurement_readout_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> &functor);
			void install_measurement_position_mean_square_error(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor);
			void install_measurement_position_frechet_distance(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor);
			void install_measurement_position_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> &functor);

			void install_performances(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &functor);
			void install_states(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor);
			void install_weights(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor);
			void install_position(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &functor);
			void install_stimulus(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &functor);

			void install_mutator(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor);
			void install_scheduler(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor);
			void install_scheduling(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor);

			void install_feedforward(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> &functor);
			void install_feedback(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> &functor);
			void install_readout(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> &functor);
			void install_recurrent(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &functor);


		protected:
			virtual void callback_completed() override;
			virtual void callback_configured(const unsigned int &id) override;
			virtual void callback_ack(const unsigned int &id, const std::size_t &number, const bool &success, const std::string &cause) override;
			virtual void callback_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name) override;
			virtual void callback_allocated(const unsigned int &id, const int &rank) override;
			virtual void callback_deallocated(const unsigned int &id, const int &rank) override;
			virtual void callback_quit(const int &rank) override;
			virtual void callback_trained(const unsigned int &id) override;
			virtual void callback_primed(const unsigned int &id) override;
			virtual void callback_tested(const unsigned int &id) override;
			virtual void callback_error(const std::string &message) override;
			virtual void callback_information(const std::string &message) override;
			virtual void callback_warning(const std::string &message) override;


			virtual void callback_measurement_readout_mean_square_error(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_readout_frechet_distance(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_readout_custom(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_position_mean_square_error(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_position_frechet_distance(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_position_custom(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override;

			virtual void callback_performances(const unsigned int &id, const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds) override;
			virtual void callback_states(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override;
			virtual void callback_weights(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override;
			virtual void callback_position(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols) override;
			virtual void callback_stimulus(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols) override;

			virtual void callback_mutator(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations) override;
			virtual void callback_scheduler(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations) override;
			virtual void callback_scheduling(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations) override;

			virtual void callback_feedforward(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_feedback(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_readout(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_recurrent(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols) override;
		public:
			static std::shared_ptr<Frontend> create(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const std::shared_ptr<TRN::Engine::Executor> &to_caller);
		};
	};
};