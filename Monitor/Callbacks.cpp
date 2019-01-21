#include "stdafx.h"
#include "Callbacks.h"
#include "Helper/Logger.h"
#include "TRN4CPP/Search.h"


static const std::string FILENAME_TOKEN = "FILENAME";
static const std::string WIDTH_TOKEN = "WIDTH";
static const std::string HEIGHT_TOKEN = "HEIGHT";
static const std::string THICKNESS_TOKEN = "THICKNESS";

struct ID
{
	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;
	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	ID(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)
	{
		TRN4CPP::Simulation::decode(simulation_id,frontend_number, condition_number, batch_number);
		TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);
	}
};

static std::ostream &operator << (std::ostream &stream, const ID id)
{
	stream << "frontend #" << id.frontend_number << ", condition #" << id.condition_number << ", batch #" << id.batch_number << ", trial #" << id.trial_number << ", train #" << id.train_number << ", test #" << id.test_number << ", repeat #" << id.repeat_number;
	return stream;
}

void Callbacks::initialize(const std::map<std::string, std::string> &arguments)
{
	TRACE_LOGGER;
}

void Callbacks::uninitialize()
{
	TRACE_LOGGER;
}

void Callbacks::callback_measurement_readout_raw(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER << __FUNCTION__ << " : " << ID(simulation_id, evaluation_id);
}

void Callbacks::callback_measurement_position_raw(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER << __FUNCTION__ << " : " << ID(simulation_id, evaluation_id);
}
void Callbacks::callback_measurement_readout_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER << __FUNCTION__ << " : " << ID(simulation_id, evaluation_id);
}
void Callbacks::callback_measurement_readout_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER << __FUNCTION__ << " : " << ID(simulation_id, evaluation_id) << ", discrete Fréchet distance = " << TRN4CPP::Search::score(values);
}
void Callbacks::callback_measurement_position_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER << __FUNCTION__ << " : " << ID(simulation_id, evaluation_id);
}
void Callbacks::callback_measurement_position_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER << __FUNCTION__ << " : " << ID(simulation_id, evaluation_id) << ", discrete Fréchet distance = " << TRN4CPP::Search::score(values);
}
void Callbacks::callback_performances(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER <<    __FUNCTION__ << " : " << ID(simulation_id, evaluation_id) << ", phase = " << phase << ", speed = " << std::fixed << cycles_per_second << " Hz, throughput = " << std::fixed << gflops_per_second << " GFlops/s" ;
}
void Callbacks::callback_states(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER <<  __FUNCTION__ << " : " << ID(simulation_id, evaluation_id) << ", phase = " << phase << ", batch = " << batch;
}
void Callbacks::callback_weights(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER << __FUNCTION__ << " : " << ID(simulation_id, evaluation_id) << ", weights = " << label << ", rows = " << rows << ", cols = " << cols ;
}
void Callbacks::callback_scheduling(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER << __FUNCTION__ << " : " << ID(simulation_id, evaluation_id);
}


void Callbacks::callback_results(const unsigned short &condition_number, const std::size_t &generation_number, const std::vector<std::pair<std::map<std::string, std::string>, std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>>> &results)
{
	TRACE_LOGGER;
	std::size_t configuration = 0;
	std::vector<std::string> strings;
	std::vector<float> avg_v;
	std::vector<float> stddev_v;
	std::vector<float> score_v;
	for (auto result : results)
	{
		std::stringstream ssr;
		std::vector<std::string> parameters;
		for (auto variable : result.first)
		{
			auto key = variable.first;
			auto value = variable.second;

			parameters.push_back(key + " = " + value);
		}
		ssr << "Configuration #" << configuration << " (" << boost::algorithm::join(parameters, ", ") << "), generation #" << generation_number << " ";
		auto prefix = ssr.str();
		for (auto by_trial : result.second)
		{
			for (auto by_train : by_trial.second)
			{
				for (auto by_test : by_train.second)
				{
					std::stringstream ssrtt;
					float avg, stddev;
					float score = by_test.second.first;
					auto measurements = by_test.second.second;



					score_v.push_back(score);

					ssrtt << prefix << "trial #" << by_trial.first << ", test #" << by_test.first << ", score = " << score;
					strings.push_back(ssr.str());
				}
			}
		}

		configuration++;
	}

	// initialize original index locations
	std::vector<size_t> idx(strings.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(),
		[&score_v](const size_t &i1, const size_t &i2)
	{
		return score_v[i1] < score_v[i2];
	});

	std::vector<std::string> rows;

	std::transform(idx.begin(), idx.end(), std::back_inserter(rows),
		[strings, score_v](const std::size_t &i)
	{
		return strings[i] + " : " + std::to_string(score_v[i]);
	});

	INFORMATION_LOGGER << __FUNCTION__ << " : " << "condition #"  << condition_number << ", results = " << std::endl << boost::algorithm::join(rows, "\n");
}


void Callbacks::callback_solutions(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions)
{
	TRACE_LOGGER;

	std::size_t configuration = 0;
	std::vector<std::string> rows;
	for (auto solution : solutions)
	{
		std::stringstream ssr;
		std::vector<std::string> parameters;
		for (auto variable : solution.first)
		{
			auto key = variable.first;
			auto value = variable.second;

			parameters.push_back(key + " = " + value);
		}
		ssr << "Configuration #" << configuration << " (" << boost::algorithm::join(parameters, ", ") << ") -> " << solution.second;
		rows.push_back(ssr.str());
		configuration++;
	}

	INFORMATION_LOGGER << __FUNCTION__ << " : " << "condition #" << condition_number << ", solutions = " << std::endl << boost::algorithm::join(rows, "\n");
}