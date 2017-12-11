#include "stdafx.h"
#include "Basic.h"
#include "Search.h"
#include "Callbacks.h"
#include "Extended.h"
#include "Helper/Logger.h"



struct Solution
{
	std::map<std::string, std::string> variables; // genotype
	std::vector<float> score;

	Solution() 
	{

	}

	Solution(const std::map<std::string, std::string> &variables) : variables(variables)
	{

	}
};

struct Population
{
	std::size_t generation;
	unsigned int simulation_number;
	std::size_t batch_number;
	std::size_t batch_size;
	std::vector<Solution> population;
};



static std::map <unsigned short, Population> pool;
static boost::shared_ptr<TRN4CPP::Plugin::Search::Interface> search;
static std::recursive_mutex mutex;

const std::string TARGET_TOKEN = "TARGET";

class Hook : public TRN4CPP::Plugin::Callbacks::Interface
{
	enum Target
	{
		READOUT_MEAN_SQUARE_ERROR,
		POSITION_MEAN_SQUARE_ERROR,
		READOUT_FRECHET_DISTANCE,
		POSITION_FRECHET_DISTANCE
	};

private:
	Target target;

public:
	virtual void initialize(const std::map<std::string, std::string> &arguments) override
	{
		if (arguments.find(TARGET_TOKEN) == arguments.end())
			throw std::invalid_argument("Target argument must be specified");
		auto token = arguments.at(TARGET_TOKEN);
		boost::to_upper(token);
		if (token == "READOUT_MEAN_SQUARE_ERROR")
			target = READOUT_MEAN_SQUARE_ERROR;
		else if (token == "POSITION_MEAN_SQUARE_ERROR")
			target = POSITION_MEAN_SQUARE_ERROR;
		else if (token == "READOUT_FRECHET_DISTANCE")
			target = READOUT_FRECHET_DISTANCE;
		else if (token == "POSITION_FRECHET_DISTANCE")
			target = POSITION_FRECHET_DISTANCE;
		else
			throw std::invalid_argument("Unexpected target " + token);

	}
	virtual void uninitialize() override
	{
	}

	virtual	void callback_measurement_readout_raw(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override
	{

	}
	virtual void callback_measurement_position_raw(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override
	{
	}
	virtual void callback_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override
	{
		if (target == READOUT_MEAN_SQUARE_ERROR)
		{
			TRN4CPP::Search::evaluate(id, values);
		}
	}
	virtual void callback_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override
	{
		if (target == READOUT_FRECHET_DISTANCE)
		{
			TRN4CPP::Search::evaluate(id, values);
		}
	}
	virtual void callback_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override
	{
		if (target == POSITION_MEAN_SQUARE_ERROR)
		{
			TRN4CPP::Search::evaluate(id, values);
		}
	}
	virtual void callback_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override
	{
		if (target == POSITION_FRECHET_DISTANCE)
		{
			TRN4CPP::Search::evaluate(id, values);
		}
	}

	virtual void callback_performances(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second) override
	{
	}
	virtual void callback_states(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override
	{
	}
	virtual void callback_weights(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override
	{
	}
	virtual void callback_scheduling(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations) override
	{
	}
};
void TRN4CPP::Plugin::Search::initialize(const std::string &library_path, const std::string &name, const std::map<std::string, std::string>  &arguments)
{
	if (search)
		throw std::runtime_error("Search plugin is already initialized");
	boost::filesystem::path path = library_path;

	path /= name;

	search = boost::dll::import<TRN4CPP::Plugin::Search::Interface>(path, "plugin_search", boost::dll::load_mode::append_decorations);
	search->initialize(arguments);
	search->install_generation(std::bind(&TRN4CPP::Search::populate, std::placeholders::_1, std::placeholders::_2));
	INFORMATION_LOGGER << "Search plugin " << name << " loaded from path " << library_path;
	auto hook = boost::make_shared<Hook>();
	hook->initialize(arguments);
	TRN4CPP::Plugin::Callbacks::append(hook);
}

void TRN4CPP::Plugin::Search::uninitialize()
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (search)
	{
		search->uninitialize();
		pool.clear();
		search.reset();
	}
}


std::map<std::string, std::set<std::string>> TRN4CPP::Search::parse(const std::string &filename)
{
	INFORMATION_LOGGER << "Reading file " << filename;
	auto extension = boost::filesystem::extension(filename);
	boost::to_upper(extension);

	boost::property_tree::iptree properties;
	std::string prefix = "";
	if (extension == ".XML")
	{
		boost::property_tree::read_xml(filename, properties);
		prefix = "<xmlattr>.";
	}
	else if (extension == ".INFO")
		boost::property_tree::read_info(filename, properties);
	else if (extension == ".INI")
		boost::property_tree::read_ini(filename, properties);
	else if (extension == ".JSON")
		boost::property_tree::read_json(filename, properties);
	else
		throw std::invalid_argument("Unexpected file extension \"" + extension + "\"");

	static const std::string variables_name = "variables";
	static const std::string variable_name = "variable";
	static const std::string value_name = "value";
	static const std::string range_name = "range";
	static const std::string name_attribute = prefix+"name";
	static const std::string begin_attribute = prefix + "begin";
	static const std::string end_attribute = prefix + "end";
	static const std::string step_attribute = prefix + "step";

	std::map<std::string, std::set<std::string>> variables;
	for (auto property_element : properties)
	{
		if (boost::iequals(property_element.first, variables_name))
		{
			auto _variables = property_element.second;

			for (auto variables_element : _variables)
			{
				if (boost::iequals(variables_element.first, variable_name))
				{
					auto _variable = variables_element.second;
					auto name = _variable.get_child(name_attribute).get_value<std::string>();
					for (auto _variable_element : _variable)
					{
						if (boost::iequals(_variable_element.first, value_name))
						{
							auto _value = _variable_element.second;

							variables[name].insert(_value.get_value<std::string>());
						}
						else if (boost::iequals(_variable_element.first, range_name))
						{
							auto _range = _variable_element.second;

							auto begin = _range.get_child(begin_attribute).get_value<float>();
							auto end = _range.get_child(end_attribute).get_value<float>();
							auto step = _range.get_child(step_attribute).get_value<float>();
							for (float value = begin; value <= end; value +=step)
							{
								variables[name].insert(std::to_string(value));
							}
							variables[name].insert(std::to_string(end));
						}
					}
				}
			}
		}
	}

	return variables;
}

std::string TRN4CPP::Search::retrieve(const unsigned short &condition_number, const unsigned int &individual_number, const std::string &key)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);

	if (individual_number >= size(condition_number))
		throw std::runtime_error("Condition #" + std::to_string(condition_number) + " does not hold invididual #" + std::to_string(individual_number));
	
	auto &variables = pool[condition_number].population[condition_number].variables;
	auto it = variables.find(key);
	if (it == variables.end())
		throw std::runtime_error("Variable " + key + " is not defined for condition #" + std::to_string(condition_number) + ", invididual #" + std::to_string(individual_number));
	return it->second;
}

unsigned int TRN4CPP::Search::size(const unsigned short &condition_number)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (pool.find(condition_number) == pool.end())
	{
		throw std::runtime_error("Condition #" + std::to_string(condition_number) + " is not populated");
	}
	if (!search)
	{
		return pool[condition_number].batch_number;
	}
	else
	{
		return pool[condition_number].population.size() * pool[condition_number].batch_number;
	}
}
void  TRN4CPP::Search::begin(const unsigned short &condition_number, const unsigned int &simulation_number, const std::size_t &batch_number, const std::size_t &batch_size)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (pool.find(condition_number) != pool.end())
	{
		throw std::runtime_error("Condition #" + std::to_string(condition_number) + " is already populated");
	}
	pool[condition_number].simulation_number = simulation_number;
	pool[condition_number].batch_number = batch_number;
	pool[condition_number].batch_size = batch_size;
	pool[condition_number].generation = 0;

	if (search)
	{
		search->callback_generation(condition_number, {});
	}
}
bool TRN4CPP_EXPORT		TRN4CPP::Search::end(const unsigned short &condition_number)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (!search)
	{
		return true;
	}
	else
	{
		TRN4CPP::Engine::Execution::run();
		if (pool.find(condition_number) == pool.end())
		{
			throw std::runtime_error("Condition #" + std::to_string(condition_number) + " is not populated");
		}
		return pool[condition_number].population.empty();
	}
}

void TRN4CPP::Search::populate(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &individuals)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);

	pool[condition_number].population.resize(individuals.size());
	
	pool[condition_number].generation++;
	DEBUG_LOGGER << "Condition number #" << condition_number << ", populating generation #" << pool[condition_number].generation;
	unsigned int number = 0;
	for (auto individual : individuals)
	{
		pool[condition_number].population[number] = Solution(individual);
		number++;
	}
}

static float aggregate(const std::vector<float> &score)
{
	float aggregated;
	float new_m, old_m = score[0];
	float new_s, old_s = 0.0f;

	for (std::size_t k = 1; k < score.size(); k++)
	{
		new_m = old_m + (score[k] - old_m) / k;
		new_s = old_s + (score[k] - old_m)*(score[k] - new_m);

		old_m = new_m;
		old_s = new_s;
	}

	float mean = score.size() > 0 ? new_m : 0.0f;
	float variance = score.size() > 1 ? new_s / (score.size() - 1) : 0.0f;
	float stddev = sqrtf(variance);
	
	return mean * stddev;
}

void TRN4CPP::Search::evaluate(const unsigned long long &id, const std::vector<float> &score)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (score.empty())
		throw std::invalid_argument("Score is empty");

	unsigned short frontend;
	unsigned short condition;
	unsigned int simulation;
	TRN4CPP::Simulation::decode(id, frontend, condition, simulation);


	unsigned int individual = (simulation - pool[condition].simulation_number) / pool[condition].batch_number;
	auto &target = pool[condition].population[individual].score;

	target.insert(target.begin(), score.begin(), score.end());

	if (std::all_of(pool[condition].population.begin(), pool[condition].population.end(), [&](const Solution &solution)
		{
		return solution.score.size() == pool[condition].batch_number * pool[condition].batch_size;
		}))
	{
		DEBUG_LOGGER << "Condition number #" << condition << ", generation #" <<  pool[condition].generation << " is evaluated";
		std::vector<float> evaluated(pool[condition].population.size());

		std::transform(pool[condition].population.begin(), pool[condition].population.end(), evaluated.begin(), [](const Solution &individual) 
			{
				return aggregate(individual.score);
			});
		search->callback_generation(condition, evaluated);
	}
}
