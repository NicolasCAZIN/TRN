#include "stdafx.h"
#include "Search.h"
#include "Helper/Logger.h"

const std::string FILENAME_TOKEN = "FILENAME";
struct Search::Handle
{
	std::map<std::string, std::set<std::string>> variables;
	std::function<void(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population)> populate;
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
	handle->variables=TRN4CPP::Search::parse(filename);
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

void Search::callback_generation(const unsigned short &condition_number, const std::vector<float> &score)
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

	cartesian_product(product,  values, current);

	std::vector<std::map<std::string, std::string>> combinations;

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

	handle->populate(condition_number, combinations);
}
void Search::install_generation(const std::function<void(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population)> &functor)
{  
	handle->populate = functor;
}


