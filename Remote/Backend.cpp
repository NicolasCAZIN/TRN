#include "stdafx.h"

#include "Model/Driver.h"
#include "Model/Simulator.h"
#include "Model/Reservoir.h"
#include "Model/Loop.h"
#include "Model/Initializer.h"
#include "Model/Scheduler.h"
#include "Network/Manager.h"

#include "Backend_impl.h"
#include "Network/Connection_impl.h"



TRN::Remote::Backend::Backend(const std::string &host, const unsigned short &port) :
	TRN::Network::Connection(TRN::Network::Manager::create()),
	handle(std::make_unique<Handle>())

{
	boost::asio::ip::tcp::resolver resolver(*TRN::Network::Connection::handle->manager);
	boost::asio::ip::tcp::resolver::iterator iterator = resolver.resolve({ host, std::to_string(port) });

	//start_connect(iterator);
	TRN::Network::Connection::handle->socket.connect(*iterator);
	//TRN::Network::Connection::handle->deadline_timer.async_wait(boost::bind(&TRN::Remote::Backend::check_deadline, this));

	initialize();
	handle->run_thread = std::thread([this]() {	this->TRN::Network::Connection::handle->manager->run(); });
	
	std::cout << "end" << std::endl;
	//io_service.run


}

TRN::Remote::Backend::~Backend()
{
	uninitialize();
	
	handle.reset();
}
template <typename Type>
static Type shift(std::queue<std::string, std::deque<std::string>> &command)
{
	if (command.empty())
		throw std::invalid_argument("command queue is empty");
	auto token = boost::lexical_cast<Type>(command.front());
	command.pop();
	return token;
}


void  TRN::Remote::Backend::start_connect(boost::asio::ip::tcp::resolver::iterator iterator)
{
	if (iterator != boost::asio::ip::tcp::resolver::iterator())
	{
		std::cout << "Trying " << iterator->endpoint() << "...\n";

		// Set a deadline for the connect operation.
		TRN::Network::Connection::handle->deadline_timer.expires_from_now(boost::posix_time::seconds(1));

		// Start the asynchronous connect operation.
		TRN::Network::Connection::handle->socket.async_connect(iterator->endpoint(),
			boost::bind(&TRN::Remote::Backend::handle_connect,
				this, _1, iterator));
	}
	else
	{
		// There are no more endpoints to try. Shut down the client.
		stop();
	}
}

void TRN::Remote::Backend::handle_connect(const boost::system::error_code& ec,
	boost::asio::ip::tcp::resolver::iterator endpoint_iter)
{
	if (handle->stopped)
		return;

	// The async_connect() function automatically opens the socket at the start
	// of the asynchronous operation. If the socket is closed at this time then
	// the timeout handler must have run first.
	if (!TRN::Network::Connection::handle->socket.is_open())
	{
		std::cout << "Connect timed out\n";

		// Try the next available endpoint.
		start_connect(++endpoint_iter);
	}

	// Check if the connect operation failed before the deadline expired.
	else if (ec)
	{
		std::cout << "Connect error: " << ec.message() << "\n";

		// We need to close the socket used in the previous connection attempt
		// before starting a new one.
		TRN::Network::Connection::handle->socket.close();

		// Try the next available endpoint.
		start_connect(++endpoint_iter);
	}

	// Otherwise we have successfully established a connection.
	else
	{
		std::cout << "Connected to " << endpoint_iter->endpoint() << "\n";

		initialize();
	}
}

void TRN::Remote::Backend::check_deadline()
{
	if (handle->stopped)
		return;

	// Check whether the deadline has passed. We compare the deadline against
	// the current time since a new asynchronous operation may have moved the
	// deadline before this actor had a chance to run.
	if (TRN::Network::Connection::handle->deadline_timer.expires_at() <= boost::asio::deadline_timer::traits_type::now())
	{
		// The deadline has passed. The socket is closed so that any outstanding
		// asynchronous operations are cancelled.
		TRN::Network::Connection::handle->socket.close();

		// There is no longer an active deadline. The expiry is set to positive
		// infinity so that the actor takes no action until a new deadline is set.
		TRN::Network::Connection::handle->deadline_timer.expires_at(boost::posix_time::pos_infin);
	}

	// Put the actor back to sleep.
	TRN::Network::Connection::handle->deadline_timer.async_wait(boost::bind(&TRN::Remote::Backend::check_deadline, this));
}

void TRN::Remote::Backend::receive_scheduling(const unsigned int &id, const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)
{
	throw std::invalid_argument("Unexpected scheduling");
}



void TRN::Remote::Backend::receive_command(const unsigned int &id, const std::vector<std::string> &command)
{
	try
	{
		std::cout << "BACKEND received : " << boost::algorithm::join(command, " ") << std::endl;
		std::queue<std::string, std::deque<std::string>> pending(std::deque<std::string>(command.begin(), command.end()));

		auto target = shift<std::string>(pending);
		auto rows = shift<std::size_t>(pending);
		auto cols = shift<std::size_t>(pending);
		if (target == "READOUT")
		{
			if (handle->readout.find(id) == handle->readout.end())
				throw std::invalid_argument("Readout functor is not defined for simulator #" + std::to_string(id));
			handle->readout[id](rows, cols);
		}
		else if (target == "RECURRENT")
		{
			if (handle->recurrent.find(id) == handle->recurrent.end())
				throw std::invalid_argument("Performances functor is not defined for simulator #" + std::to_string(id));
			handle->recurrent[id](rows, cols);
		}
		else if (target == "FEEDFORWARD")
		{
			if (handle->feedforward.find(id) == handle->feedforward.end())
				throw std::invalid_argument("Feedforward functor is not defined for simulator #" + std::to_string(id));
			handle->feedforward[id](rows, cols);
		}
		else if (target == "FEEDBACK")
		{
			if (handle->feedback.find(id) == handle->feedback.end())
				throw std::invalid_argument("Feedback functor is not defined for simulator #" + std::to_string(id));
			handle->feedback[id](rows, cols);
		}
		else
			throw std::invalid_argument("Unexpected command " + target);
		if (!pending.empty())
		{
			throw std::invalid_argument("Pending unprocessed tokens in command : " + boost::algorithm::join(command, " "));
		}
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}
void TRN::Remote::Backend::receive_matrix(const unsigned int &id, const std::string &label, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)
{
	if (elements.size() != rows * cols)
	{
		throw std::invalid_argument("data size != rows * cols");
	}

	std::vector<std::string> tokens;

	boost::split(tokens, label, boost::is_any_of("\t "));
	std::queue<std::string, std::deque<std::string>> pending(std::deque<std::string>(tokens.begin(), tokens.end()));
	auto type = shift<std::string>(pending);

	if (type == "STIMULUS")
	{
		if (rows != 1)
			throw std::invalid_argument("prediction must be a column vector");
		if (handle->predicted_stimulus.find(id) == handle->predicted_stimulus.end())
			throw std::invalid_argument("Prediction functor is not defined for simulator #" + std::to_string(id));
		handle->predicted_stimulus[id](elements);
	}
	else if (type == "POSITION")
	{
		if (rows != 1)
			throw std::invalid_argument("position must be a column vector");
		if (handle->predicted_position.find(id) == handle->predicted_position.end())
			throw std::invalid_argument("Posuition functor is not defined for simulator #" + std::to_string(id));
		handle->predicted_position[id](elements);
	}
	else if (type == "STATES")
	{
		auto name = shift<std::string>(pending);
		if (handle->states.find(id) == handle->states.end()) 
			throw std::invalid_argument("States functor is not defined for simulator #" + std::to_string(id));
		handle->states[id](name, elements, rows, cols);
	}
	else if (type == "WEIGHTS")
	{
		auto name = shift<std::string>(pending);
		if (handle->weights.find(id) == handle->weights.end())
			throw std::invalid_argument("Weights functor is not defined for simulator #" + std::to_string(id));
		handle->weights[id](name, elements, rows, cols);
	}
	else if (type == "PERFORMANCES")
	{
		auto stage = shift<std::string>(pending);
		if (rows != 1 || cols != 1)
			throw std::invalid_argument("Performances data must be a scalar");
		if (handle->performances.find(id) == handle->performances.end())
			throw std::invalid_argument("Performances functor is not defined for simulator #" + std::to_string(id));
		handle->performances[id](stage, elements[0]);
	}
	else
	{
		throw std::invalid_argument("Unexpected label " + label);
	}
}
void TRN::Remote::Backend::allocate(const unsigned int &id)
{
	send(id, { "ALLOCATE" });
}
void TRN::Remote::Backend::deallocate(const unsigned int &id)
{
	send(id, { "DEALLOCATE" });
}
void TRN::Remote::Backend::train(const unsigned int &id, const std::string &label, const std::string &incoming, const std::string &expected)
{
	send(id, { "TRAIN", label, incoming, expected });
}
void TRN::Remote::Backend::test(const unsigned int &id, const std::string &label, const std::string &incoming, const std::string &expected, const int &preamble)
{
	send(id, { "TEST", label, incoming, expected, std::to_string(preamble) });
}
void TRN::Remote::Backend::declare_sequence(const unsigned int &id, const std::string &label, const std::string &tag,
	const std::vector<float> &sequence, const std::size_t &observations)
{
	send(id, { "DECLARE", label, tag });
	send(id, "SEQUENCE", sequence, observations, sequence.size() / observations);
}
void TRN::Remote::Backend::declare_batch(const unsigned int &id, const std::string &label, const std::string &tag,
	const std::vector<std::string> &labels)
{
	std::vector<std::string> to_send = { "DECLARE", label, tag };
	to_send.insert(to_send.begin(), labels.begin(), labels.end());
	send(id, to_send);
}
void TRN::Remote::Backend::setup_states(const unsigned int &id, const std::function<void(const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (handle->states.find(id) != handle->states.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a weights functor");
	handle->states[id]= functor;
	send(id, { "SETUP", "STATES"});
}
void TRN::Remote::Backend::setup_weights(const unsigned int &id, const std::function<void(const std::string &label, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (handle->weights.find(id) != handle->weights.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a weights functor");
	handle->weights[id] = functor;
	send(id, { "SETUP", "WEIGHTS"});
}

void TRN::Remote::Backend::setup_performances(const unsigned int &id, const std::function<void(const std::string &phase, const float &cycles_per_second)> &functor)
{
	if (handle->performances.find(id) != handle->performances.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a performances functor");
	handle->performances[id] = functor;
	send(id, { "SETUP", "PERFORMANCES"});
}
void TRN::Remote::Backend::configure_begin(const unsigned int &id)
{
	send(id, { "CONFIGURE", "BEGIN"});
}
void TRN::Remote::Backend::configure_end(const unsigned int &id)
{
	send(id, { "CONFIGURE", "END"});
}

void TRN::Remote::Backend::configure_reservoir_widrow_hoff(const unsigned int &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate,
	const float &initial_state_scale, const float &learning_rate)
{
	send(id, { "CONFIGURE", "RESERVOIR", "WIDROW-HOFF", std::to_string(stimulus_size), std::to_string(prediction_size), std::to_string(reservoir_size),std::to_string(leak_rate),std::to_string(initial_state_scale),std::to_string(learning_rate) });
}

void TRN::Remote::Backend::configure_loop_copy(const unsigned int &id, const std::size_t &stimulus_size)
{
	send(id, { "CONFIGURE", "LOOP", "COPY", std::to_string(stimulus_size) });
}
void TRN::Remote::Backend::configure_loop_spatial_filter(const unsigned int &id, const std::size_t &stimulus_size,
	const std::function<void(const std::vector<float> &position)> &predicted_position,
	std::function<void(const std::vector<float> &position)> &estimated_position,
	const std::function<void(const std::vector<float> &position)> &predicted_stimulus,
	std::function<void(const std::vector<float> &stimulus)> &perceived_stimulus,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &x, const std::pair<float, float> &y,
	const std::vector<float> &response,
	const float &sigma,
	const float &radius,
	const std::string &tag)
{
	if (handle->predicted_position.find(id) != handle->predicted_position.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a position prediction functor");
	handle->predicted_position[id] = predicted_position;
	if (handle->predicted_stimulus.find(id) != handle->predicted_stimulus.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a stimulus prediction functor");
	handle->predicted_stimulus[id] = predicted_stimulus;
	estimated_position = [this, id](const std::vector<float> &position)
	{
		send(id, { "POSITION" }, position, 1, position.size());
	};
	perceived_stimulus = [this, id](const std::vector<float> &stimulus)
	{
		send(id, { "STIMULUS" }, stimulus, 1, stimulus.size());
	};
	send(id, "FIRING_RATE_MAP", response, rows * stimulus_size, cols);
	send(id, { "CONFIGURE", "LOOP", "SPATIAL_FILTER", std::to_string(stimulus_size),  std::to_string(rows) ,  std::to_string(cols),
		std::to_string(x.first),
		std::to_string(x.second),
		std::to_string(y.first),
		std::to_string(y.second),
		std::to_string(sigma),
		std::to_string(radius),
		tag
	});
}
void TRN::Remote::Backend::configure_loop_custom(const unsigned int &id, const std::size_t &stimulus_size,
	const std::function<void(const std::vector<float> &prediction)> &request,
	std::function<void(const std::vector<float> &stimulus)> &reply)
{
	if (handle->predicted_stimulus.find(id) != handle->predicted_stimulus.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a prediction functor");
	handle->predicted_stimulus[id] = request;
	reply = [this, id](const std::vector<float> &stimulus) 
	{
		send(id, "INPUT", stimulus, 1, stimulus.size());
	};
	send(id, { "CONFIGURE", "LOOP", "CUSTOM", std::to_string(stimulus_size) });
}

void TRN::Remote::Backend::configure_scheduler_tiled(const unsigned int &id, const int &epochs)
{
	send(id, { "CONFIGURE", "SCHEDULER", "TILED",  std::to_string(epochs) });
}
void TRN::Remote::Backend::configure_scheduler_snippets(const unsigned int &id, const int &snippets_size, const int &time_budget, const std::string &tag)
{
	send(id, { "CONFIGURE", "SCHEDULER", "SNIPPETS",  std::to_string(snippets_size), std::to_string(time_budget),  tag });
}
void TRN::Remote::Backend::configure_scheduler_custom(const unsigned int &id,
	const std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &reply, const std::string &tag)
{
	if (handle->scheduler.find(id) != handle->scheduler.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a scheduler functor");
	handle->scheduler[id] = request;
	reply = [this, id](const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		send(id, offsets, durations);
	};
	send(id, { "CONFIGURE", "SCHEDULER", "CUSTOM", tag });
}

void TRN::Remote::Backend::configure_readout_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	send(id, { "CONFIGURE", "WEIGHTS", "UNIFORM", "READOUT", std::to_string(a), std::to_string(b), std::to_string(sparsity) });
}
void TRN::Remote::Backend::configure_readout_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	send(id, { "CONFIGURE", "WEIGHTS", "GAUSSIAN",  "READOUT", std::to_string(mu), std::to_string(sigma) });
}
void TRN::Remote::Backend::configure_readout_custom(const unsigned int &id, 
	const std::function<void(const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (handle->readout.find(id) != handle->readout.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a readout functor");;
	handle->readout[id] = request;
	reply = [this, id](const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)
	{
		send(id, "READOUT", weights, rows, cols);
	};
	send(id, { "CONFIGURE", "WEIGHTS", "CUSTOM", "READOUT" });
}

void TRN::Remote::Backend::configure_feedback_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	send(id, { "CONFIGURE", "WEIGHTS", "UNIFORM", "FEEDBACK", std::to_string(a), std::to_string(b), std::to_string(sparsity) });
}
void TRN::Remote::Backend::configure_feedback_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	send(id, { "CONFIGURE", "WEIGHTS", "GAUSSIAN",  "FEEDBACK", std::to_string(mu), std::to_string(sigma) });
}
void TRN::Remote::Backend::configure_feedback_custom(const unsigned int &id,
	const std::function<void(const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (handle->feedback.find(id) != handle->feedback.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a feedback functor");
	handle->feedback[id] = request;
	reply = [this, id](const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)
	{
		send(id, "FEEDBACK", weights, rows, cols);
	};
	send(id, { "CONFIGURE", "WEIGHTS", "CUSTOM", "FEEDBACK" });
}
void TRN::Remote::Backend::configure_recurrent_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	send(id, { "CONFIGURE", "WEIGHTS", "UNIFORM", "RECURRENT",std::to_string(a), std::to_string(b), std::to_string(sparsity) });
}
void TRN::Remote::Backend::configure_recurrent_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	send(id, { "CONFIGURE", "WEIGHTS", "GAUSSIAN",  "RECURRENT",  std::to_string(mu), std::to_string(sigma) });
}
void TRN::Remote::Backend::configure_recurrent_custom(const unsigned int &id,
	const std::function<void(const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (handle->recurrent.find(id) != handle->recurrent.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a recurrent functor");
	handle->recurrent[id] = request;
	reply = [this, id](const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)
	{
		send(id, "RECURRENT", weights, rows, cols);
	};
	send(id, { "CONFIGURE", "WEIGHTS", "CUSTOM", "RECURRENT" });
}

void TRN::Remote::Backend::configure_feedforward_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	send(id, { "CONFIGURE", "WEIGHTS", "UNIFORM", "FEEDFORWARD", std::to_string(a), std::to_string(b), std::to_string(sparsity) });
}
void TRN::Remote::Backend::configure_feedforward_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	send(id, { "CONFIGURE", "WEIGHTS", "GAUSSIAN",  "FEEDFORWARD", std::to_string(mu), std::to_string(sigma) });
}
void TRN::Remote::Backend::configure_feedforward_custom(const unsigned int &id,
	const std::function<void(const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (handle->feedforward.find(id) != handle->feedforward.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a feedforward functor");
	handle->feedforward[id] = request;
	reply = [this, id](const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)
	{
		send(id, "FEEDFORWARD", weights, rows, cols);
	};
	send(id, { "CONFIGURE", "WEIGHTS", "CUSTOM", "FEEDFORWARD" });
}

std::shared_ptr<TRN::Remote::Backend> TRN::Remote::Backend::create(const std::string &host, const unsigned short &port)
{
	return std::make_shared<TRN::Remote::Backend>(host, port);
}