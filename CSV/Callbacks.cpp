#include "stdafx.h"
#include "Callbacks.h"
#include "Helper/Queue.h"
#include "Helper/Logger.h"

static const std::string DEFAULT_IDENTIFIER = "";
static const std::string IDENTIFIER_TOKEN = "IDENTIFIER";
static const std::string FILENAME_TOKEN = "FILENAME";
static const std::string DEFAULT_SEPARATOR = ",";
static const std::string SEPARATOR_TOKEN = "SEPARATOR";

static const char *PERFORMANCES_FIELD = "performances";
static const char *STATES_FIELD = "states";
static const char *WEIGHTS_FIELD = "weights";
static const char *SCHEDULING_FIELD = "scheduling";
static const char *RECORDING_FIELD = "recording";

static const char *MEAN_SQUARE_ERROR_FIELD = "mean_square_error";
static const char *RAW_FIELD = "raw";
static const char *FRECHET_DISTANCE_FIELD = "frechet_distance";
static const char *MEASUREMENT_FIELD = "measurement";
static const char *READOUT_FIELD = "readout";
static const char *POSITION_FIELD = "position";

class Callbacks::Handle
{
public :
	std::string identifier;
	std::string filename;
	std::string separator;
};

void Callbacks::initialize(const std::map<std::string, std::string> &arguments)
{
	TRACE_LOGGER;
	if (handle)
		throw std::runtime_error("Handle is already allocated");
	handle = std::make_unique<Handle>();
	if (arguments.find(FILENAME_TOKEN) == arguments.end())
		throw std::runtime_error("Can't file " + FILENAME_TOKEN + " key/value pair");
	else
		handle->filename = arguments.at(FILENAME_TOKEN);
	if (arguments.find(IDENTIFIER_TOKEN) == arguments.end())
		handle->identifier = DEFAULT_IDENTIFIER;
	else
		handle->identifier = arguments.at(IDENTIFIER_TOKEN);
	if (arguments.find(SEPARATOR_TOKEN) == arguments.end())
		handle->separator = DEFAULT_SEPARATOR;
	else
		handle->separator = arguments.at(SEPARATOR_TOKEN);
}
void Callbacks::uninitialize()
{
	handle.reset();
}

 void Callbacks::dump_csv(const unsigned long long &simulation_id, const unsigned long long &evaluation_id,
	const std::vector<std::string> &path, const std::vector<std::vector<std::string>> &values)
{
	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;

	TRN4CPP::Simulation::decode(simulation_id, frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);


	boost::filesystem::path filename_path(handle->filename);
	auto basename = boost::filesystem::basename(filename_path);
	auto extension = boost::filesystem::extension(filename_path);
	auto parent_directory = filename_path.parent_path();

	std::vector<std::string> filename_parts;

	filename_parts.push_back(basename);
	filename_parts.push_back(handle->identifier);

	for (auto part : path)
		filename_parts.push_back(part);

	filename_parts.push_back("frontend");
	filename_parts.push_back(std::to_string(frontend_number));
	filename_parts.push_back("condition");
	filename_parts.push_back(std::to_string(condition_number));
	filename_parts.push_back("batch");
	filename_parts.push_back(std::to_string(batch_number));
	filename_parts.push_back("trial");
	filename_parts.push_back(std::to_string(trial_number));
	filename_parts.push_back("train");
	filename_parts.push_back(std::to_string(train_number));
	filename_parts.push_back("test");
	filename_parts.push_back(std::to_string(test_number));
	filename_parts.push_back("repeat");
	filename_parts.push_back(std::to_string(repeat_number));
	
	auto filename = boost::algorithm::join(filename_parts, "_") + extension;
	auto absolute_filename = (parent_directory / filename).string();

	std::ofstream file;
	try
	{
		file.open(absolute_filename);

		for (auto row : values)
		{
			file << boost::algorithm::join(row, handle->separator) << std::endl;
		}
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what();
	}
	if (file.is_open())
	{
		file.close();
	}
}

 void Callbacks::dump_matrix(const unsigned long long &simulation_id, const unsigned long long &evaluation_id,
	const std::vector<std::string> &path, const std::size_t &rows, const std::size_t &cols, const std::vector<float> &elements)
{
	std::vector<std::vector<std::string>> values(rows + 1);

	std::vector<std::string> header;
	if (cols == 2)
	{
		header.push_back("x");
		header.push_back("y");
	}
	else
	{
		for (std::size_t col = 0; col < cols; col++)
		{
			header.push_back("col" + std::to_string(col));
		}
	}
	values[0] = header;
	for (std::size_t row = 0; row < rows; row++)
	{
		std::vector<std::string> tuple(cols);

		for (std::size_t col = 0; col < cols; col++)
		{
			tuple[col] = std::to_string(elements[row * cols + col]);
		}
		values[row + 1] = tuple;
	}

	dump_csv(simulation_id, evaluation_id, path, values);
}




void Callbacks::callback_measurement_readout_raw(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;

	dump_matrix(simulation_id, evaluation_id, { "measurement", "raw" , "readout", "primed" }, preamble, cols, primed);
	dump_matrix(simulation_id, evaluation_id, { "measurement", "raw" , "readout", "expected" },  rows, cols , expected);
	dump_matrix(simulation_id, evaluation_id, { "measurement", "raw" , "readout", "predicted" },  rows, cols , predicted);
}
void Callbacks::callback_measurement_position_raw(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;

	dump_matrix(simulation_id, evaluation_id, { "measurement", "raw" , "position", "primed" }, preamble, cols, primed);
	dump_matrix(simulation_id, evaluation_id, { "measurement", "raw" , "position", "expected" },  rows, cols, expected);
	dump_matrix(simulation_id, evaluation_id, { "measurement", "raw" , "position", "predicted" }, rows, cols, predicted);
}
void Callbacks::callback_measurement_readout_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;

	dump_matrix(simulation_id, evaluation_id, { "measurement", "mean_square_error" , "readout"}, rows, cols , values);
}
void Callbacks::callback_measurement_readout_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	
	dump_matrix(simulation_id, evaluation_id, { "measurement", "frechet_distance" , "readout" }, rows, cols, values);
}
void Callbacks::callback_measurement_position_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;

	dump_matrix(simulation_id, evaluation_id, { "measurement", "mean_square_error" , "position" }, rows, cols, values);
}
void Callbacks::callback_measurement_position_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;

	dump_matrix(simulation_id, evaluation_id, { "measurement", "frechet_distance" , "position" }, rows, cols, values);
}

void Callbacks::callback_performances(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	TRACE_LOGGER;
	
	dump_csv(simulation_id, evaluation_id, { "recording", "performances" }, 
		{
			{"phase", "cycles_per_second", "gflops_per_second"},
			{phase, std::to_string(cycles_per_second), std::to_string(gflops_per_second)}
		});
}
void Callbacks::callback_states(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	throw std::runtime_error("Not yet implemented");
	

}
void Callbacks::callback_weights(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	throw std::runtime_error("Not yet implemented");
	


}
void Callbacks::callback_scheduling(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;

	std::vector<std::string> offsets_vector;
	std::vector<std::string> durations_vector;
	auto conv = [](int i) { return std::to_string(i); };
	std::transform(offsets.begin(), offsets.end(), std::back_inserter(offsets_vector), conv);
	std::transform(offsets.begin(), offsets.end(), std::back_inserter(offsets_vector), conv);

	dump_csv(simulation_id, evaluation_id, { "recording", "scheduling", "offsets" },
		{
			{"offsets"},
			offsets_vector
		});
	dump_csv(simulation_id, evaluation_id, { "recording", "scheduling", "durations" },
		{
			{"durations"},
			offsets_vector
		});
}


void Callbacks::callback_results(const unsigned short &condition_number, const std::size_t &generation_number, const std::vector<std::pair<std::map<std::string, std::string>, std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>>> &results)
{
	throw std::runtime_error("Not yet implemented");
	/*std::size_t configuration_number = 0;
	for (auto result : results)
	{
		for (auto by_trial : result.second)
		{
			for (auto by_train : by_trial.second)
			{
				for (auto by_test : by_train.second)
				{
					auto &score = by_test.second.first;
					auto &measurements = by_test.second.second;

					std::map<std::string, mxArray *> results_map;
					auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; results_map["condition"] = mx_condition;
					auto mx_generation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_generation) = generation_number; results_map["generation"] = mx_generation;

					for (auto variable : result.first)
					{
						auto key = variable.first;
						auto value = variable.second;
						auto mx_value = mxCreateString(value.c_str()); results_map[key] = mx_value;
					}

					auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = by_trial.first; results_map["trial"] = mx_trial;
					auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = by_train.first; results_map["train"] = mx_train;
					auto mx_test = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_test) = by_test.first; results_map["test"] = mx_test;
					auto mx_measurements = mxCreateNumericMatrix(measurements.size(), 1, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(measurements.begin(), measurements.end(), (float *)mxGetData(mx_measurements)); results_map["measurements"] = mx_measurements;
					auto mx_score = mxCreateNumericMatrix(1, 1, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); *(float *)mxGetData(mx_score) = score; results_map["score"] = mx_score;
					auto mx_configuration = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_configuration) = configuration_number; results_map["configuration"] = mx_configuration;
					append(handle->result, { "search", "population" }, results_map);
					update();
				}
			}
		}
		configuration_number++;
	}*/
}

void Callbacks::callback_solutions(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions)
{
	throw std::runtime_error("Not yet implemented");
	/*std::vector<std::string> rows;
	for (auto solution : solutions)
	{
		std::map<std::string, mxArray *> solutions_map;
		auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; solutions_map["condition"] = mx_condition;

		for (auto variable : solution.first)
		{
			auto key = variable.first;
			auto value = variable.second;
			auto mx_value = mxCreateString(value.c_str()); solutions_map[key] = mx_value;
		}
		auto mx_score = mxCreateNumericMatrix(1, 1, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); *(float *)mxGetData(mx_score) = solution.second; solutions_map["score"] = mx_score;
		append(handle->result, { "search", "solutions" }, solutions_map);
		update();
	}*/
}