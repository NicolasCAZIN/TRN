#include "stdafx.h"
#include "Search.h"
#include "Helper/Logger.h"


static const std::size_t DEFAULT_TEST = 0;
static const std::size_t DEFAULT_TRIAL = 0;
static const std::size_t DEFAULT_TRAIN = 0;
static const std::string TEST_TOKEN = "TEST";
static const std::string TRAIN_TOKEN = "TRAIN";
static const std::string TRIAL_TOKEN = "TRIAL";

static const std::size_t DEFAULT_POPULATION = 10;
static const std::size_t DEFAULT_ITERATIONS = 100;
static const std::size_t DEFAULT_REPEAT = 10;
static const float DEFAULT_T0 = 1000.0f;
static const float DEFAULT_TN = 1e-20f;
static const float DEFAULT_ALPHA = 0.01f;
static const unsigned long long DEFAULT_SEED = 0;
static const std::string SEED_TOKEN = "SEED";
static const std::string FILENAME_TOKEN = "FILENAME";

static const std::string ALPHA_TOKEN = "ALPHA";
static const std::string OBJECTIVE_TOKEN = "OBJECTIVE";
static const std::string POPULATION_TOKEN = "POPULATION";
static const std::string ITERATIONS_TOKEN = "ITERATIONS";
static const std::string REPEAT_TOKEN = "REPEAT";
static const std::string T0_TOKEN = "T0";
static const std::string TN_TOKEN = "TN";
struct Search::Handle
{
	float T0;
	float alpha;
	float alpha_t;
	std::size_t repeat;
	std::size_t max_repeat;
	std::size_t iteration;
	std::size_t max_iterations;
	std::vector<std::map<std::string, std::string>> population;
	std::vector<std::map<std::string, std::string>> solution;

	
	std::function<float()>  rng;
	std::size_t population_size;
	std::size_t test;
	std::size_t trial;
	std::size_t train;
	std::map<std::string, std::vector<std::string>> variables;
	std::vector<std::string> labels;
	std::vector<float> cost;
	float best_cost;
	std::map<std::string, std::string> best;

	std::function<void(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population)> populate;
	std::function<void(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions) > publish;
};



void Search::initialize(const std::map<std::string, std::string> &arguments)
{
	if (handle)
		throw std::runtime_error("Handle already allocated");
	handle = std::make_unique<Handle>();
	std::string prefix = "";

	if (arguments.find(FILENAME_TOKEN) == arguments.end())
		throw std::runtime_error("Can't file " + FILENAME_TOKEN + " key/value pair");
	auto filename = arguments.at(FILENAME_TOKEN);
	auto variables = TRN4CPP::Search::parse(filename);
	for (auto variable : variables)
	{
		auto key = variable.first;
		std::vector<std::string> v;
		std::copy(variable.second.begin(), variable.second.end(), std::back_inserter(v));
		handle->variables[key] = v;
		handle->labels.push_back(key);
	}
	if (arguments.find(TEST_TOKEN) == arguments.end())
		handle->test = DEFAULT_TEST;
	else
		handle->test = boost::lexical_cast<std::size_t>(arguments.at(TEST_TOKEN));
	if (handle->test == DEFAULT_TRIAL)
	{
		INFORMATION_LOGGER << "Costs from all tests will be aggreagated into one";
	}
	else
	{
		INFORMATION_LOGGER << "Test #" << handle->test << " will be used for evaluating cost";
	}


	if (arguments.find(TRAIN_TOKEN) == arguments.end())
		handle->train = DEFAULT_TRAIN;
	else
		handle->train = boost::lexical_cast<std::size_t>(arguments.at(TRAIN_TOKEN));
	if (handle->train == DEFAULT_TRAIN)
	{
		INFORMATION_LOGGER << "Costs from all train will be aggreagated into one";
	}
	else
	{
		INFORMATION_LOGGER << "Train #" << handle->train << " will be used for evaluating cost";
	}


	if (arguments.find(TRIAL_TOKEN) == arguments.end())
		handle->trial = DEFAULT_TRIAL;
	else
		handle->trial = boost::lexical_cast<std::size_t>(arguments.at(TRIAL_TOKEN));

	if (handle->trial == DEFAULT_TRIAL)
	{
		INFORMATION_LOGGER << "Costs from all trials will be aggreagated into one";
	}
	else
	{
		INFORMATION_LOGGER << "Trial #" << handle->trial << " will be used for evaluating cost";
	}

	if (arguments.find(POPULATION_TOKEN) == arguments.end())
		handle->population_size = DEFAULT_POPULATION;
	else
		handle->population_size = boost::lexical_cast<std::size_t>(arguments.at(POPULATION_TOKEN));

	if (arguments.find(T0_TOKEN) == arguments.end())
		handle->T0 = DEFAULT_T0;
	else
		handle->T0 = boost::lexical_cast<float>(arguments.at(T0_TOKEN));
	float TN;
	if (arguments.find(TN_TOKEN) == arguments.end())
		TN = DEFAULT_TN;
	else
		TN = boost::lexical_cast<float>(arguments.at(TN_TOKEN));


	handle->alpha_t = 1.0f;
	if (arguments.find(ITERATIONS_TOKEN) == arguments.end())
		handle->max_iterations = DEFAULT_ITERATIONS;
	else
		handle->max_iterations = boost::lexical_cast<std::size_t>(arguments.at(ITERATIONS_TOKEN));
	handle->alpha = std::powf((float)(TN / handle->T0), 1.0f / ((float)handle->max_iterations));
	INFORMATION_LOGGER << "T0 = " << handle->T0 << ", TN = " << TN << ", alpha = " << handle->alpha;
	handle->iteration = 0;
	if (arguments.find(REPEAT_TOKEN) == arguments.end())
		handle->max_repeat = DEFAULT_REPEAT;
	else
		handle->max_repeat = boost::lexical_cast<std::size_t>(arguments.at(REPEAT_TOKEN));
	unsigned long long seed;
	if (arguments.find(SEED_TOKEN) == arguments.end())
		seed = DEFAULT_SEED;
	else
		seed = boost::lexical_cast<std::size_t>(arguments.at(SEED_TOKEN));


	std::uniform_real<float> distribution(0.0f, 1.0f);
	std::mt19937_64 generator(seed);

	handle->rng = std::bind(distribution, generator);

	handle->cost.resize(handle->population_size);
	handle->population.resize(handle->population_size);
	handle->solution.resize(handle->population_size);


	INFORMATION_LOGGER << "Population size = " << handle->population_size << ", seed = " << seed;
}
void Search::uninitialize()
{
	handle.reset();
}



std::size_t Search::select(const std::size_t &size)
{
	return (std::size_t)std::floor(handle->rng() * (size - 1));
}

std::string Search::choose(const std::vector<std::string> &possible)
{
	return possible[select(possible.size())];
}
std::map<std::string, std::string> Search::neighboor(const std::map<std::string, std::string> &candidate)
{
	auto altered = candidate;
	std::string chosen;
	do
	{
		chosen = choose(handle->labels);
	} while (handle->variables[chosen].size() <= 1);
	auto possible = handle->variables[chosen];

	do
	{
		altered[chosen] = choose(possible);
	} while (altered[chosen] == candidate.at(chosen));

	return altered;
}

static std::string explain(const std::map<std::string, std::string> &individual, float &cost)
{
	std::vector<std::string> variables;
	for (auto variable : individual)
	{
		auto key = variable.first;
		auto value = variable.second;

		variables.push_back(key + " = " + value);
	}
	return boost::join(variables, ", ") + ", cost = " + std::to_string(cost);
}

void Search::reset()
{
	handle->population.clear();
	handle->solution.clear();

	handle->best.clear();
	handle->best_cost = std::numeric_limits<float>::max();
	std::fill(handle->cost.begin(), handle->cost.end(), std::numeric_limits<float>::max());
	handle->iteration = 0;
	handle->alpha_t = 1.0f;
}

void Search::callback_generation(const unsigned short &condition_number, const std::vector<std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>> &measurements)
{
	if (measurements.empty())
	{
		if (handle->iteration != 0)
			throw std::runtime_error("Unexpected iteration number");
		INFORMATION_LOGGER << "Initializing population with random configurations";

		reset();
		handle->population.resize(handle->population_size);
		handle->solution.resize(handle->population_size);
		for (std::size_t k = 0; k < handle->population_size; k++)
		{
			for (auto variable : handle->variables)
			{
				auto key = variable.first;
				auto possible = variable.second;

				handle->population[k][key] = choose(possible);
			}
		}
	}
	else
	{
		handle->iteration++;
		auto temperature = handle->T0 * handle->alpha_t;
		INFORMATION_LOGGER << "Iteration " << handle->iteration << "/" << handle->max_iterations;
		INFORMATION_LOGGER << "Temperature " << temperature;
		if (measurements.size() != handle->population_size)
			throw std::runtime_error("Population size mismatch");
	
		for (std::size_t k = 0; k < handle->population_size; k++)
		{
			auto old_cost = handle->cost[k];
			auto new_cost = TRN4CPP::Search::evaluate_cost(measurements[k], handle->trial, handle->train, handle->test);
			auto delta_E = new_cost - old_cost;
			auto acceptance = std::expf(-delta_E / temperature);

			if (acceptance > handle->rng())
			{
				handle->solution[k] = handle->population[k];
				handle->cost[k] = new_cost;
			}

			INFORMATION_LOGGER << "Solution " << k << " : " << explain(handle->solution[k], handle->cost[k]);

			if (new_cost < handle->best_cost)
			{
				handle->best = handle->population[k];
				handle->best_cost = new_cost;
			}
		}
		
		INFORMATION_LOGGER << "Best solution so far : " << explain(handle->best, handle->best_cost);
		if (handle->iteration < handle->max_iterations)
		{
			handle->alpha_t *= handle->alpha;
			if (handle->iteration % handle->max_repeat == 0)
			{
				INFORMATION_LOGGER << "Populating with neighbors of the best solution";
				for (std::size_t k = 0; k < handle->population_size; k++)
				{
					handle->population[k] = neighboor(handle->best);
				}
			}
			else
			{
				INFORMATION_LOGGER << "Populating with neighbors of solution";
				for (std::size_t k = 0; k < handle->population_size; k++)
				{
					handle->population[k] = neighboor(handle->solution[k]);
				}
			}
		}
		else
		{
			struct Comp
			{
				bool operator()(const std::pair<std::map<std::string, std::string>, float> &lhs, const std::pair<std::map<std::string, std::string>, float> &rhs) const
				{
					return lhs.second < rhs.second;
				}
			
			};
			std::set<std::pair<std::map<std::string, std::string>, float>, Comp> solution_set;
			for (std::size_t k = 0; k < handle->solution.size(); k++)
			{
				solution_set.insert(std::make_pair(handle->solution[k], handle->cost[k]));
			}
			solution_set.insert(std::make_pair(handle->best, handle->best_cost));
			std::vector<std::pair<std::map<std::string, std::string>, float>> solutions;
			std::copy(solution_set.begin(), solution_set.end(), std::back_inserter(solutions));
		
			handle->publish(condition_number, solutions);

			INFORMATION_LOGGER << "Simulated annealing is terminated";
			reset();

		}
	
	}

	handle->populate(condition_number, handle->population);
}
void Search::install_generation(const std::function<void(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population)> &functor)
{
	handle->populate = functor;
}

void Search::install_solutions(const std::function<void(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions)> &functor)
{
	handle->publish = functor;
}
