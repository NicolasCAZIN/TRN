#include "stdafx.h"
#include "Basic.h"
#include "Search.h"
#include "Callbacks.h"
#include "Extended.h"
#include "Helper/Logger.h"

extern std::function<void(const unsigned short &condition_number, const std::size_t &generation_number, const std::vector<std::pair<std::map<std::string, std::string>, std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>>> &results)> on_search_results;
extern std::function<void(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions)> on_search_solutions;
typedef std::map < std::size_t, std::map<std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>> Score;
typedef std::map<std::string, std::string> Parameters;
template <typename Type>
static bool contiguous(const std::set<Type> &set)
{
	if (set.empty())
		return false;
	std::vector<Type> v(set.size());
	std::copy(set.begin(), set.end(), v.begin());
	//std::sort(v.begin(), v.end());
	for (std::size_t k = 0; k < v.size() - 1; k++)
	{
		if (v[k] != v[k + 1] - 1)
			return false;
	}
	return true;
}
template <typename Key, typename Value>
static std::set<Key> extract_keys(const std::map<Key, Value> &map)
{
	std::set<Key> keys;

	for (auto p : map)
	{
		keys.insert(p.first);
	}

	return keys;
}

float TRN4CPP::Search::score(const std::vector<float> &score)
{
	float aggregated;
	float new_m, old_m = score[0];
	float new_s, old_s = 0.0f;

	new_m = old_m;
	for (std::size_t k = 1; k < score.size(); k++)
	{
		new_m = old_m + (score[k] - old_m) / k;
		new_s = old_s + (score[k] - old_m)*(score[k] - new_m);

		old_m = new_m;
		old_s = new_s;
	}

	float mean = score.size() > 0 ? new_m : 0.0f;
	float variance = score.size() > 1 ? new_s / (score.size() - 1) : 0.0f;
	float stddev = std::sqrtf(variance);

	return mean + stddev;
}

struct Measurement
{
	std::size_t repeat;
	std::vector<float> score;
};

class Individual
{
	std::map<std::size_t, std::map<std::size_t, std::map<std::size_t, Measurement>>> measurements;
	Parameters parameters; // genotype

public:
	Individual()
	{

	}

	Individual(const std::map<std::string, std::string> &parameters) : parameters(parameters)
	{

	}

	std::pair<Parameters, Score> aggregate() const
	{
		Score aggregated;
	
		for (auto by_trial_number : measurements)
		{
			const std::size_t trial_number = by_trial_number.first;
			for (auto by_train_number : by_trial_number.second)
			{
				const std::size_t train_number = by_train_number.first;
				for (auto by_test_number : by_train_number.second)
				{
					std::size_t test_number = by_test_number.first;
					auto &m = by_test_number.second.score;
					aggregated[trial_number][train_number][test_number].second = m;
					aggregated[trial_number][train_number][test_number].first = TRN4CPP::Search::score(m);
				}
			}
		}

		return std::make_pair(parameters, aggregated);
	}

	void operator () (const std::size_t &trial_number, const std::size_t &train_number, const std::size_t &test_number, const std::size_t &repeat_number, const std::vector<float> &score)
	{
		auto &current = measurements[trial_number][test_number][test_number].score;
		DEBUG_LOGGER << "Inserting " << score.size() << " results for trial #" << trial_number << ", train #" << test_number << ", train #" << test_number << ", repeat #" << repeat_number;

		current.insert(current.begin(), score.begin(), score.end());
	}
	const std::string &operator[](const std::string &key)
	{
		auto it = parameters.find(key);
		if (it == parameters.end())
			throw std::runtime_error("Variable " + key + " is not defined");
		return it->second;
	}

/*private:
	void retrieve(const std::size_t &trial_number, const std::size_t &evaluation_number, std::size_t &test_number, std::size_t &repeat_number)
	{
		auto it = measurements.find(trial_number);
		if (it == measurements.end())
			throw std::runtime_error("measurement is not declared for trial : " + std::to_string(trial_number));

		std::set<std::size_t> test_number_set = extract_keys(it->second);
		if (!contiguous(test_number_set))
			throw std::runtime_error("test number set is not contiguous");
		std::size_t cummulated_repeat = 0;
		bool found = false;
		for (auto t : test_number_set)
		{
			auto r = it->second[t].repeat;
			auto reminder = (evaluation_number - 1)- cummulated_repeat;
	
			if (reminder < r)
			{
				repeat_number = reminder + 1;
				test_number = t;
				found = true;
				break;
			}
			cummulated_repeat += r;
		}
		if (!found)
			throw std::runtime_error("test_number and repeat_number were not retrieved");
	}*/
public :
	void update_repeat(const std::size_t &trial_number, const std::size_t &train_number, const std::size_t &test_number, const std::size_t &repeat)
	{
		measurements[trial_number][train_number][test_number].repeat = repeat;	
	}


	bool completed(const std::size_t &expected) const
	{
		auto trial_number_set = extract_keys(measurements);
		if (!contiguous(trial_number_set))
			return false;
		for (auto trial_number : trial_number_set)
		{
			auto &by_trial = measurements.at(trial_number);
			auto train_number_set = extract_keys(by_trial);
			if (!contiguous(train_number_set))
				return false;
			for (auto train_number : train_number_set)
			{
				auto &by_train = by_trial.at(train_number);
				auto test_number_set = extract_keys(by_train);
				if (!contiguous(test_number_set))
					return false;

				for (auto test_number : test_number_set)
				{
					auto measurement = by_train.at(test_number);
					if (measurement.repeat == 0)
						throw std::runtime_error("Repeat must be at least 1");
					auto expected_score_size = expected * measurement.repeat;

					if (measurement.score.size() != expected_score_size)
						return false;
				}
			}
		}
		return true;
	}
};

class Population
{

	std::mutex mutex;
	std::condition_variable cond;

	std::size_t generation;
	unsigned int offset;
	std::size_t bundle_size;
	std::size_t batch_size;
	
	std::vector<Individual> population;

	bool evaluated;

public :
	void initialize(const unsigned int &batch_number,
		const std::size_t &bundle_size,
		const std::size_t &batch_size)
	{
		this->bundle_size = bundle_size;
		this->batch_size = batch_size;
		this->offset = batch_number - 1;
		this->generation = 0;
		this->evaluated = false;
	
	}
	void update_offset(const unsigned int &batch_number)
	{
		this->offset = batch_number - 1;
	
	}


	std::size_t simulations()
	{
		if (population.empty())
			return bundle_size;
		else
			return population.size() * bundle_size;
	}

	Individual &operator [](const unsigned int &batch_number)
	{
		std::size_t index = ((batch_number - 1) - this->offset) / bundle_size;
		return population[index];
	}

	void renew(const std::vector<std::map<std::string, std::string>> &individuals)
	{
		population.resize(individuals.size());
	
		if (population.empty())
		{
			INFORMATION_LOGGER << "Search is over";
		}
		else
		{
			generation++;
			unsigned int number = 0;
			for (auto individual : individuals)
			{
				population[number] = Individual(individual);
				number++;
			}
			INFORMATION_LOGGER << "Populating generation #" << generation << " with " << number << " configurations";
		}

	}

	bool completed()
	{
		auto evaluated = bundle_size * batch_size;
		return std::all_of(population.begin(), population.end(), [&](const Individual &individual)
		{
			return individual.completed(evaluated);
		});
	}

	void wait(std::unique_lock<std::recursive_mutex> &recursive_mutex)
	{
		
		std::unique_lock<std::mutex> guard(mutex);
		recursive_mutex.unlock();
		while (!evaluated)
		{
			cond.wait(guard);
		}
		evaluated = false;
		recursive_mutex.lock();
	}

	void notify()
	{
		std::unique_lock<std::mutex> guard(mutex);
		evaluated = true;
		guard.unlock();
		cond.notify_one();
	}

	bool empty()
	{
		return population.empty();
	}

	std::pair<std::size_t, std::vector<std::pair<Parameters, Score>>>  results()
	{
		std::vector<std::pair<Parameters, Score>> evaluated(population.size());

		std::transform(population.begin(), population.end(), evaluated.begin(), [](const Individual &individual)
		{
			return individual.aggregate(); 
		});

		return std::make_pair(generation, evaluated);
	}
};

static std::map <unsigned short, Population> pool;
static std::recursive_mutex mutex;
static boost::shared_ptr<TRN4CPP::Plugin::Search::Interface> search;

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

	virtual	void callback_measurement_readout_raw(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override
	{

	}
	virtual void callback_measurement_position_raw(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override
	{
	}
	virtual void callback_measurement_readout_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id,  const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override
	{
		if (target == READOUT_MEAN_SQUARE_ERROR)
		{
			TRN4CPP::Search::evaluate(simulation_id,evaluation_id,values);
		}
	}
	virtual void callback_measurement_readout_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override
	{
		if (target == READOUT_FRECHET_DISTANCE)
		{
			TRN4CPP::Search::evaluate(simulation_id, evaluation_id, values);
		}
	}
	virtual void callback_measurement_position_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override
	{
		if (target == POSITION_MEAN_SQUARE_ERROR)
		{
			TRN4CPP::Search::evaluate(simulation_id, evaluation_id, values);
		}
	}
	virtual void callback_measurement_position_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override
	{
		if (target == POSITION_FRECHET_DISTANCE)
		{
			TRN4CPP::Search::evaluate(simulation_id, evaluation_id, values);
		}
	}

	virtual void callback_performances(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second) override
	{
	}
	virtual void callback_states(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch,  const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override
	{
	}
	virtual void callback_weights(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override
	{
	}
	virtual void callback_scheduling(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations) override
	{
	}
	virtual void callback_results(const unsigned short &condition_number, const std::size_t &generation_number, const std::vector<std::pair<std::map<std::string, std::string>, std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>>> &results) override
	{

			std::vector<std::map<std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>> scores;
			std::transform(results.begin(), results.end(), std::back_inserter(scores), [](const std::pair<std::map<std::string, std::string>, std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>> &result)
			{
				return result.second;
			}
			);

			search->callback_generation(condition_number, scores);
		
	}

	virtual void callback_solutions(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions) override
	{

	}

};
float TRN4CPP::Search::evaluate_cost(const  std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>> &measurements, const std::size_t &trial_number, const std::size_t &train_number, const std::size_t &test_number)
{
	std::vector<float> costs;

	std::set<std::size_t> trial_set;

	if (trial_number == 0)
	{
		for (auto m : measurements)
		{
			trial_set.insert(m.first);
		}
	}
	else
	{
		trial_set.insert(trial_number);
	}


	for (auto trial : trial_set)
	{
		std::set<std::size_t> train_set;

		if (train_number == 0)
		{
			for (auto m : measurements.at(trial))
			{
				train_set.insert(m.first);
			}
		}
		else
		{
			train_set.insert(train_number);
		}

		for (auto train : train_set)
		{
			std::set<std::size_t> test_set;

			if (test_number == 0)
			{
				for (auto m : measurements.at(trial).at(train))
				{
					test_set.insert(m.first);
				}
			}
			else
			{
				test_set.insert(test_number);
			}
			for (auto test : train_set)
			{
				costs.push_back(measurements.at(trial).at(train).at(test).first);
			}
		}
	}

	return TRN4CPP::Search::score(costs);
}

void TRN4CPP::Plugin::Search::initialize(const std::string &library_path, const std::string &name, const std::map<std::string, std::string>  &arguments)
{
	if (search)
		throw std::runtime_error("Search plugin is already initialized");
	boost::filesystem::path path = library_path;

	path /= name;

	search = boost::dll::import<TRN4CPP::Plugin::Search::Interface>(path, "plugin_search", boost::dll::load_mode::append_decorations);
	search->initialize(arguments);
	search->install_generation(TRN4CPP::Search::populate);
	search->install_solutions(on_search_solutions);
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

	std::map<std::string, std::set<std::string>> parameters;
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

							parameters[name].insert(_value.get_value<std::string>());
						}
						else if (boost::iequals(_variable_element.first, range_name))
						{
							auto _range = _variable_element.second;

							auto begin = _range.get_child(begin_attribute).get_value<float>();
							auto end = _range.get_child(end_attribute).get_value<float>();
							auto step = _range.get_child(step_attribute).get_value<float>();
							bool is_integer = std::floor(begin) == begin && std::floor(end) == end && std::floor(step) == step;
							for (auto value = begin; value <= end; value += step)
							{
								if (is_integer)
								{
									parameters[name].insert(std::to_string((int)value));
								}
								else
								{
									parameters[name].insert(std::to_string(value));
								}
								
							}
							if (is_integer)
							{
								parameters[name].insert(std::to_string((int)end));
							}
							else
							{
								parameters[name].insert(std::to_string(end));
							}
						}
					}
				}
			}
		}
	}
	for (auto variable : parameters)
	{
		INFORMATION_LOGGER << "Variable " << variable.first << " have " << variable.second.size() << " values";
	}
	return parameters;
}


std::string TRN4CPP::Search::retrieve(const unsigned short &condition_number, const unsigned int &batch_number, const std::string &key)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (pool.find(condition_number) == pool.end())
	{
		throw std::runtime_error("Condition #" + std::to_string(condition_number) + " is not populated");
	}

	return pool[condition_number][batch_number][key];
}

unsigned int TRN4CPP::Search::size(const unsigned short &condition_number)
{
	// retourned une liste de simulation incompletes
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (pool.find(condition_number) == pool.end())
	{
		throw std::runtime_error("Condition #" + std::to_string(condition_number) + " is not populated");
	}
	return pool[condition_number].simulations();

}
void  TRN4CPP::Search::begin(const unsigned short &condition_number, const unsigned int &batch_number, const std::size_t &bundle_size, const std::size_t &batch_size)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (pool.find(condition_number) != pool.end())
	{
		throw std::runtime_error("Condition #" + std::to_string(condition_number) + " is already populated");
	}
	pool[condition_number].initialize(batch_number, bundle_size, batch_size);

	if (search)
	{
		search->callback_generation(condition_number, {});
	}
}

void 		TRN4CPP::Search::update(const unsigned short &condition_number, const unsigned int &batch_number, const std::size_t &trial_number, const std::size_t &train_number, const std::size_t &test_number, const std::size_t &repeat)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);

	if (pool.find(condition_number) == pool.end())
	{
		throw std::runtime_error("Condition #" + std::to_string(condition_number) + " is not populated");
	}

	if (search)
	{
		pool[condition_number][batch_number].update_repeat(trial_number, train_number, test_number, repeat);
	}
}

bool 	TRN4CPP::Search::end(const unsigned short &condition_number, const unsigned int &batch_number)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (!search)
	{
		return true;
	}
	else
	{
		if (pool.find(condition_number) == pool.end())
		{
			throw std::runtime_error("Condition #" + std::to_string(condition_number) + " is not populated");
		}
		pool[condition_number].wait(guard);

		if (pool[condition_number].empty())
		{
			INFORMATION_LOGGER << "Condition #" << condition_number << " completed. Removing population from the pool";
			pool.erase(condition_number);
			return true;
		}
		else
		{
			pool[condition_number].update_offset(batch_number);
			return false;
		}
	}
}

void TRN4CPP::Search::populate(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &individuals)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (pool.find(condition_number) == pool.end())
	{
		throw std::runtime_error("Condition #" + std::to_string(condition_number) + " is not populated");
	}
	pool[condition_number].renew(individuals);
}

void TRN4CPP::Search::evaluate(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &measurements)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (measurements.empty())
		throw std::invalid_argument("Score is empty");

	unsigned short frontend;
	unsigned short condition;
	unsigned int batch;
	TRN4CPP::Simulation::decode(simulation_id,frontend, condition, batch);
	DEBUG_LOGGER << "Reporting results for condition #" << condition << ", batch #" << batch;

	unsigned short trial;
	unsigned short train;
	unsigned short test;
	unsigned short repeat;
	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial, train, test, repeat);


	pool[condition][batch](trial, train, test, repeat,  measurements);

	if (pool[condition].completed())
	{
		auto &p = pool[condition].results();
		auto &generation = p.first;
		auto &results = p.second;
		on_search_results(condition,  generation, results);

		pool[condition].notify();
	}
}
