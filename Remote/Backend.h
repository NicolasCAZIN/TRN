#pragma once

#include "remote_global.h"
#include "Engine/Backend.h"
#include "Helper/Proxy.h"
#include "Core/Simulator.h"
#include "Network/Connection.h"

namespace TRN
{
	namespace Remote
	{
		class REMOTE_EXPORT Backend  : 
			public TRN::Engine::Backend,
			public TRN::Network::Connection
		{
		private :
			class Handle;
			mutable std::unique_ptr<Handle> handle;

		public:
			Backend(const std::string &host, const unsigned short &port);
			~Backend();

		private :
			void receive_scheduling(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations);
			void receive_command(const unsigned int &id, const std::vector<std::string> &command);
			void receive_matrix(const unsigned int &id, const std::string &label, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols);

		public:
			virtual void 	allocate(const unsigned int &id) override;
			virtual void 	deallocate(const unsigned int &id) override;
			virtual void 	train(const unsigned int &id, const std::string &sequence, const std::string &incoming, const std::string &expected) override;
			virtual void 	test(const unsigned int &id, const std::string &sequence, const std::string &incoming, const std::string &expected, const int &preamble) override;
			virtual void 	declare_sequence(const unsigned int &id, const std::string &label, const std::string &tag,
				const std::vector<float> &sequence, const std::size_t &observations) override;
			virtual void 	declare_batch(const unsigned int &id, const std::string &label, const std::string &tag,
				const std::vector<std::string> &labels) override;
			virtual void 	setup_states(const unsigned int &id, const std::function<void(const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor) override;
			virtual void 	setup_weights(const unsigned int &id, const std::function<void(const std::string &label, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor) override;
			virtual void 	setup_performances(const unsigned int &id, const std::function<void(const std::string &phase, const float &cycles_per_second)> &functor) override;


			virtual void	configure_begin(const unsigned int &id) override;
			virtual void	configure_end(const unsigned int &id) override;
			virtual void 	configure_reservoir_widrow_hoff(const unsigned int &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate,
				const float &initial_state_scale, const float &learning_rate) override;

			virtual void 	configure_loop_copy(const unsigned int &id, const std::size_t &stimulus_size) override;
			virtual void 	configure_loop_spatial_filter(const unsigned int &id, const std::size_t &stimulus_size,
				const std::function<void(const std::vector<float> &position)> &predicted_position,
				std::function<void(const std::vector<float> &position)> &estimated_position,
				const std::function<void(const std::vector<float> &position)> &predicted_stimulus,
				std::function<void(const std::vector<float> &stimulus)> &perceived_stimulus,
				const std::size_t &rows, const std::size_t &cols,
				const std::pair<float, float> &x, const std::pair<float, float> &y,
				const std::vector<float> &response,
				const float &sigma,
				const float &radius,
				const std::string &tag
			) override;
			virtual void 	configure_loop_custom(const unsigned int &id, const std::size_t &stimulus_size,
				const std::function<void(const std::vector<float> &prediction)> &request,
				std::function<void(const std::vector<float> &stimulus)> &reply) override;

			virtual void 	configure_scheduler_tiled(const unsigned int &id, const int &epochs) override;
			virtual void 	configure_scheduler_snippets(const unsigned int &id, const int &snippets_size, const int &time_budget, const std::string &tag) override;
			virtual void 	configure_scheduler_custom(const unsigned int &id
				const std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
				std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &reply, const std::string &tag) override;

			virtual void 	configure_readout_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity) override;
			virtual void 	configure_readout_gaussian(const unsigned int &id, const float &mu, const float &sigma) override;
			virtual void 	configure_readout_custom(const unsigned int &id
				const std::function<void(const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &reply) override;

			virtual void 	configure_feedback_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity) override;
			virtual void 	configure_feedback_gaussian(const unsigned int &id, const float &mu, const float &sigma) override;
			virtual void 	configure_feedback_custom(const unsigned int &id
				const std::function<void(const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &reply) override;

			virtual void 	configure_recurrent_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity) override;
			virtual void 	configure_recurrent_gaussian(const unsigned int &id, const float &mu, const float &sigma) override;
			virtual void 	configure_recurrent_custom(const unsigned int &id
				const std::function<void(const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &reply) override;

			virtual void 	configure_feedforward_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity) override;
			virtual void 	configure_feedforward_gaussian(const unsigned int &id, const float &mu, const float &sigma) override;
			virtual void 	configure_feedforward_custom(const unsigned int &id
				const std::function<void(const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &reply) override;
		private :
			void handle_connect(const boost::system::error_code& ec,
				boost::asio::ip::tcp::resolver::iterator endpoint_iter);
			void start_connect(boost::asio::ip::tcp::resolver::iterator iterator);
			void check_deadline();

		public:
			static std::shared_ptr<Backend> create(const std::string &host, const unsigned short &port);
		};
	};
};