#include "stdafx.h"
#include "Custom.h"
#include "Helper/Logger.h"
#include "Helper/Parser.h"

static const std::string FILENAME_TOKEN = "FILENAME";
static const std::string SEED_TOKEN = "SEED";
static const std::string RADIUS_THRESHOLD_TOKEN = "RADIUS_THRESHOLD";
static const std::string JITTER_TOKEN = "JITTER";

struct Custom::Handle
{
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> notify_position;
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> notify_stimulus;
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> notify_mutator;
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> notify_scheduler;
	std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> notify_feedforward;
	std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> notify_feedback;
	std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> notify_recurrent;
	std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> notify_readout;

	unsigned long seed;
	float jitter;
	std::pair<float, float> arena_x;
	std::pair<float, float> arena_y;
	std::vector<float> x_center;
	std::vector<float> y_center;
	std::vector<float> K;
	std::size_t place_cells;
};

void Custom::initialize(const std::map<std::string, std::string> &arguments)
{
	if (handle)
		throw std::runtime_error("Handle is already initialized");

	handle = std::make_unique<Handle>();
	if (arguments.find(FILENAME_TOKEN) == arguments.end())
		throw std::invalid_argument("Can't find argument " + FILENAME_TOKEN);
	if (arguments.find(SEED_TOKEN) == arguments.end())
		throw std::invalid_argument("Can't find argument " + SEED_TOKEN);
	if (arguments.find(JITTER_TOKEN) == arguments.end())
		throw std::invalid_argument("Can't find argument " + JITTER_TOKEN);
	if (arguments.find(RADIUS_THRESHOLD_TOKEN) == arguments.end())
		throw std::invalid_argument("Can't find argument " + RADIUS_THRESHOLD_TOKEN);
	auto filename = arguments.at(FILENAME_TOKEN);
	handle->seed = std::stoul(arguments.at(SEED_TOKEN));
	handle->jitter = std::stof(arguments.at(JITTER_TOKEN));
	auto radius_threshold = std::stof(arguments.at(RADIUS_THRESHOLD_TOKEN));
	TRN::Helper::Parser::place_cells_model(filename, radius_threshold, handle->x_center, handle->y_center, handle->K);


	auto xi = std::minmax_element(handle->x_center.begin(), handle->x_center.end());
	auto yi = std::minmax_element(handle->y_center.begin(), handle->y_center.end());
	handle->arena_x.first = *xi.first;
	handle->arena_x.second = *xi.second;
	handle->arena_y.first = *yi.first;
	handle->arena_y.second = *yi.second;
	handle->place_cells = handle->K.size();
}
void Custom::uninitialize()
{
	handle.reset();
}
void Custom::callback_position(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
{
	std::uniform_real_distribution<float> x_jitter(handle->jitter * handle->arena_x.first, handle->jitter * handle->arena_x.second);
	std::uniform_real_distribution<float> y_jitter(handle->jitter * handle->arena_y.first, handle->jitter * handle->arena_y.second);

	if (!handle->notify_position)
		throw std::runtime_error("Position notify functor is not installed");
	if (!handle->notify_stimulus)
		throw std::runtime_error("Stimulus notify functor is not installed");

	if (cols != 2)
		throw std::invalid_argument("Dimensionnality for a xy position must be 2");
	std::vector<float> jittered_position(rows * cols);
	std::vector<float> activation_pattern(rows * handle->place_cells);
#pragma omp parallel for
	for (int row = 0; row < rows; row++)
	{
		std::default_random_engine rng(handle->seed + row);
		auto x = boost::algorithm::clamp(position[row * cols + 0] + x_jitter(rng), handle->arena_x.first, handle->arena_x.second);
		auto y = boost::algorithm::clamp(position[row * cols + 1] + y_jitter(rng), handle->arena_y.first, handle->arena_y.second);
		for (std::size_t place_cell = 0; place_cell < handle->place_cells; place_cell++)
		{
			auto dx = x - handle->x_center[place_cell];
			auto dy = y - handle->y_center[place_cell];

			activation_pattern[row * handle->place_cells + place_cell] = std::expf((dx*dx + dy*dy)* handle->K[place_cell]);
		}

		jittered_position[row * cols + 0] = x;
		jittered_position[row * cols + 1] = y;

	}
	handle->seed += rows;
	handle->notify_position(simulation_id, evaluation_id, jittered_position, rows, cols);
	handle->notify_stimulus(simulation_id, evaluation_id, activation_pattern, rows, handle->place_cells);
}
void Custom::install_position(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (handle->notify_position)
		throw std::runtime_error("Position notify functor is already installed");
	handle->notify_position = functor;
}

void Custom::callback_stimulus(const unsigned long long &simulation_id, const unsigned long long &evaluation_id,  const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
{
	if (!handle->notify_stimulus)
		throw std::runtime_error("Stimulus notify functor is not installed");
}
void Custom::install_stimulus(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (handle->notify_stimulus)
		throw std::runtime_error("Position notify functor is already installed");
	handle->notify_stimulus = functor;
}

void Custom::callback_mutator(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed,  const std::vector<int> &offsets, const std::vector<int> &durations)
{
	throw std::runtime_error("Not implemented");
}
void Custom::install_mutator(const std::function<void(const unsigned long long &simulation_id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor){}

void Custom::callback_scheduler(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed,  const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	throw std::runtime_error("Not implemented");
}
void Custom::install_scheduler(const std::function<void(const unsigned long long &simulation_id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor){}

void Custom::callback_feedforward(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	throw std::runtime_error("Not implemented");
}
void Custom::install_feedforward(const std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &functor){}


void Custom::callback_readout(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	throw std::runtime_error("Not implemented");
}
void Custom::install_readout(const std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &functor){}

void Custom::callback_recurrent(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	throw std::runtime_error("Not implemented");
}
void Custom::install_recurrent(const std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &functor){}
