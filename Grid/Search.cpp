#include "stdafx.h"
#include "Search.h"
#include "Helper/Logger.h"

const std::string FILENAME_TOKEN = "FILENAME";
const std::string OBJECTIVE_TOKEN = "OBJECTIVE";

enum Objective
{
	SCORE,
	MEAN
};
static const Objective DEFAULT_OBJECTIVE = SCORE;
static const std::size_t DEFAULT_TEST = 1;
static const std::size_t DEFAULT_TRIAL = 1;
static const std::string TEST_TOKEN = "TEST";
static const std::string TRIAL_TOKEN = "TRIAL";

struct Search::Handle
{
	Objective objective;
	std::size_t trial;
	std::size_t test;
	std::vector<std::map<std::string, std::string>> population;
	std::map<std::string, std::set<std::string>> variables;
	std::function<void(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population)> populate;
	std::function<void(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions) > publish;
};

std::istream & operator >> (std::istream &is, Objective &objective)
{
	std::string token;
	is >> token;

	boost::to_upper(token);
	if (token == "SCORE")
		objective = SCORE;
	else if (token == "MEAN")
		objective = MEAN;
	else
		throw std::invalid_argument("Unexpected token " + token);

	return is;
}

void Search::initialize(const std::map<std::string, std::string> &arguments)
{
	if (handle)
		throw std::runtime_error("Handle already allocated");
	handle = std::make_unique<Handle>();
	std::string prefix = "";

	if (arguments.find(FILENAME_TOKEN) == arguments.end())
		throw std::runtime_error("Can't file " + FILENAME_TOKEN + " key/value pair");
	auto filename = arguments.at(FILENAME_TOKEN);
	handle->variables=TRN4CPP::Search::parse(filename);

	if (arguments.find(TEST_TOKEN) == arguments.end())
		handle->test = DEFAULT_TEST;
	else
		handle->test = boost::lexical_cast<std::size_t>(arguments.at(TEST_TOKEN));
	if (arguments.find(TRIAL_TOKEN) == arguments.end())
		handle->trial = DEFAULT_TRIAL;
	else
		handle->trial = boost::lexical_cast<std::size_t>(arguments.at(TRIAL_TOKEN));
	if (arguments.find(OBJECTIVE_TOKEN) == arguments.end())
		handle->objective = DEFAULT_OBJECTIVE;
	else
		handle->objective = boost::lexical_cast<Objective>(arguments.at(OBJECTIVE_TOKEN));
}
void Search::uninitialize()
{
	handle.reset();
}

static void cartesian_product(std::vector<std::vector<std::string>> &cartesian,
	std::vector<std::set<std::string>> &values,
	std::vector<std::string> &current, int col = 0)
{
	if (col < values.size())
	{
		for (auto value : values[col])
		{
			current[col] = value;
			cartesian_product(cartesian, values, current, col + 1);
		}
	}
	else
	{
		cartesian.push_back(current);
	}
}



void Search::callback_generation(const unsigned short &condition_number, const std::vector<std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>> &measurements)
{
	std::vector<std::map<std::string, std::string>> combinations;
	if (measurements.empty())
	{
		std::vector<std::set<std::string>> values;
		std::vector<std::string> keys;
		for (auto variable : handle->variables)
		{
			keys.push_back(variable.first);
			values.push_back(variable.second);
		}
		std::vector<std::vector<std::string>> product;
		std::vector<std::string> current(values.size());

		cartesian_product(product, values, current);

		for (auto tuple : product)
		{
			int k = 0;
			std::map<std::string, std::string> combination;
			for (auto value : tuple)
			{
				auto key = keys[k];
				combination[key] = value;
				k++;
			}
			combinations.push_back(combination);
		}
		handle->population = combinations;
	}
	else
	{
		std::vector<std::pair<std::map<std::string, std::string>, float>> solutions;

		for (std::size_t k = 0; k < handle->population.size(); k++)
		{
			solutions.push_back(std::make_pair(handle->population[k], TRN4CPP::Search::evaluate_cost(measurements[k])));
		}

		handle->publish(condition_number, solutions);
	}

	handle->populate(condition_number, combinations);
}
void Search::install_generation(const std::function<void(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population)> &functor)
{  
	handle->populate = functor;
}


void Search::install_solutions(const std::function<void(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions)> &functor)
{
	handle->publish = functor;
}