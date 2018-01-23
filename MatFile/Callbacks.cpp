#include "stdafx.h"
#include "Callbacks.h"
#include "Helper/Queue.h"
#include "Helper/Logger.h"
static const std::string DEFAULT_MODE = "w7.3";
static const std::string DEFAULT_IDENTIFIER = "";
static const std::string IDENTIFIER_TOKEN = "IDENTIFIER";
static const std::string TIMEOUT_TOKEN = "TIMEOUT";
static const std::string FILENAME_TOKEN = "FILENAME";
static const std::string MODE_TOKEN = "MODE";

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

struct Callbacks::Handle
{
	std::clock_t timestamp;
	std::mutex version;
	std::size_t version_saved;
	std::size_t version_updated;
	float timeout;
	std::string filename;
	std::string mode;
	std::string identifier;
	mxArray *result;
	std::thread dump;
	TRN::Helper::Queue<std::pair<std::size_t, mxArray *>> to_save;
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
	if (arguments.find(TIMEOUT_TOKEN) == arguments.end())
		throw std::runtime_error("Can't file " + TIMEOUT_TOKEN + " key/value pair");
	else
		handle->timeout = boost::lexical_cast<float>(arguments.at(TIMEOUT_TOKEN));
	if (arguments.find(MODE_TOKEN) == arguments.end())
		handle->mode = DEFAULT_MODE;
	else
		handle->mode = arguments.at(MODE_TOKEN);
	if (arguments.find(IDENTIFIER_TOKEN) == arguments.end())
		handle->identifier = DEFAULT_IDENTIFIER;
	else
		handle->identifier = arguments.at(IDENTIFIER_TOKEN);
	handle->version_saved = 0;
	handle->result = mxCreateStructMatrix(1, 1, 0, NULL);

	handle->dump = std::thread([&]() 
	{
		std::pair<std::size_t, mxArray *> p;
		while (handle->to_save.dequeue(p))
		{
			auto version = p.first;
			auto to_save = p.second;
			std::unique_lock<std::mutex> lock(handle->version);
			save(version, to_save);
			lock.unlock();
			mxDestroyArray(to_save);
		}
	});
	handle->timestamp = std::clock();
}
void Callbacks::uninitialize()
{	
	handle->to_save.invalidate();
	if (handle->dump.joinable())
		handle->dump.join();
	save(handle->version_updated, handle->result);
	mxDestroyArray(handle->result);
	handle.reset();
}

void Callbacks::save(const std::size_t &version, mxArray *result)
{
	TRACE_LOGGER;
	if (version > handle->version_saved)
	{
		try
		{
			boost::filesystem::path path(handle->filename);

			auto basename = boost::filesystem::basename(path);
			auto extension = boost::filesystem::extension(path);
			auto parent_directory = path.parent_path();

			auto filename = basename + "_" + handle->identifier + "_" + std::to_string(version) + extension;
			auto absolute_filename = (parent_directory / filename).string();
			INFORMATION_LOGGER <<   "saving to file " << absolute_filename ;

			auto pmat = matOpen(absolute_filename.c_str(), handle->mode.c_str());
			if (pmat == NULL)
				throw std::runtime_error("Can't open Mat-file " + absolute_filename + " in mode " + handle->mode);

			if (matPutVariable(pmat, "result", result))
				throw std::runtime_error("Can't put variable result");
			if (matClose(pmat))
				throw std::runtime_error("Can't close matfile");
			handle->version_saved = version;	
		}
		catch (std::exception &e)
		{
			ERROR_LOGGER << e.what() ;
		}
	}
}

void Callbacks::update()
{
	TRACE_LOGGER;
	handle->version_updated++;
	auto timestamp = std::clock();
	auto elapsed = ( timestamp - handle->timestamp) / (float)CLOCKS_PER_SEC;
	if (elapsed >= handle->timeout)
	{
		std::unique_lock<std::mutex> lock(handle->version, std::defer_lock);

		if (lock.try_lock())
		{
			if (handle->version_updated > handle->version_saved)
			{
				INFORMATION_LOGGER << "Version " << handle->version_updated << " will be saved";
				auto to_save = handle->result;
				handle->result = mxCreateStructMatrix(1, 1, 0, NULL);
				handle->to_save.enqueue(std::make_pair(handle->version_updated, to_save));
			}
		}
		else
		{
			WARNING_LOGGER << "file " << handle->filename << " is still being saved. Waiting for the next timeout" ;
		}
	
		handle->timestamp = std::clock();
	}
}

static void append(mxArray *root, const std::vector<std::string> &path, const std::map<std::string, mxArray *> &fields)
{
	mxArray *node = root;
	for (auto field : path)
	{
		boost::to_lower(field);
		auto field_str = field.c_str();
		auto child = mxGetField(node, 0, field_str);
		if (!child)
		{
			auto field_number = mxAddField(node, field_str);
			if (field_number == -1)
				throw std::runtime_error("Can't add field " + field);
			child = mxCreateStructMatrix(1, 1, 0, NULL);
			mxSetFieldByNumber(node, 0, field_number, child);
		}
		node = child;
	}

	if (mxGetNumberOfFields(node) == 0)
	{
		for (auto field : fields)
		{
			auto field_number = mxAddField(node, field.first.c_str());

			mxSetFieldByNumber(node, 0, field_number, field.second);
		}
	}
	else
	{
		auto m = mxGetM(node);
		auto size = m + 1;
		mxSetData(node, mxRealloc(mxGetData(node), size * fields.size() * sizeof(mxArray *)));
		mxSetM(node, size);

		for (auto field : fields)
		{
			mxSetField(node, m, field.first.c_str(), field.second);
		};
	}
}

void Callbacks::callback_measurement_readout_raw(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	std::size_t predicted_size[3] = { cols, rows, pages };
	std::size_t primed_size[2] = { cols, preamble };
	std::size_t expected_size[2] = { cols, rows };
	std::map<std::string, mxArray *> measurement;

	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;
	
	TRN4CPP::Simulation::decode(simulation_id,frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_batch) = batch_number; measurement["batch"] = mx_batch;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial_number;  measurement["trial"] = mx_trial;
	auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = train_number;  measurement["train"] = mx_train;
	auto mx_test = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_test) = test_number;  measurement["test"] = mx_test;
	auto mx_repeat = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_repeat) = repeat_number;  measurement["repeat"] = mx_repeat;

	auto mx_primed = mxCreateNumericArray(2, primed_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(primed.begin(), primed.end(), (float *)mxGetData(mx_primed)); measurement["primed"] = mx_primed;
	auto mx_predicted = mxCreateNumericArray(3, predicted_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(predicted.begin(), predicted.end(), (float *)mxGetData(mx_predicted)); measurement["predicted"] = mx_predicted;
	auto mx_expected = mxCreateNumericArray(2, expected_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(expected.begin(), expected.end(), (float *)mxGetData(mx_expected)); measurement["expected"] = mx_expected;
	
	append(handle->result, { "measurement", "raw" , "readout"}, measurement);

	update();
}
void Callbacks::callback_measurement_position_raw(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	std::size_t predicted_size[3] = { cols, rows, pages };
	std::size_t primed_size[2] = { cols, preamble };
	std::size_t expected_size[2] = { cols, rows };
	std::map<std::string, mxArray *> measurement;

	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;

	TRN4CPP::Simulation::decode(simulation_id, frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_batch) = batch_number; measurement["batch"] = mx_batch;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial_number;  measurement["trial"] = mx_trial;
	auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = train_number;  measurement["train"] = mx_train;
	auto mx_test = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_test) = test_number;  measurement["test"] = mx_test;
	auto mx_repeat = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_repeat) = repeat_number;  measurement["repeat"] = mx_repeat;

	auto mx_primed = mxCreateNumericArray(2, primed_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(primed.begin(), primed.end(), (float *)mxGetData(mx_primed)); measurement["primed"] = mx_primed;
	auto mx_predicted = mxCreateNumericArray(3, predicted_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(predicted.begin(), predicted.end(), (float *)mxGetData(mx_predicted)); measurement["predicted"] = mx_predicted;
	auto mx_expected = mxCreateNumericArray(2, expected_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(expected.begin(), expected.end(), (float *)mxGetData(mx_expected)); measurement["expected"] = mx_expected;

	append(handle->result, { "measurement", "raw" , "position" }, measurement);

	update();
}
void Callbacks::callback_measurement_readout_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	std::map<std::string, mxArray *> measurement;

	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;

	TRN4CPP::Simulation::decode(simulation_id, frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_batch) = batch_number; measurement["batch"] = mx_batch;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial_number;  measurement["trial"] = mx_trial;
	auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = train_number;  measurement["train"] = mx_train;
	auto mx_test = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_test) = test_number;  measurement["test"] = mx_test;
	auto mx_repeat = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_repeat) = repeat_number;  measurement["repeat"] = mx_repeat;

	auto mx_values = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(values.begin(), values.end(), (float *)mxGetData(mx_values)); measurement["values"] = mx_values;

	append(handle->result, { "measurement", "mean_square_error" , "readout" }, measurement);

	update();
}
void Callbacks::callback_measurement_readout_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	std::map<std::string, mxArray *> measurement;

	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;

	TRN4CPP::Simulation::decode(simulation_id, frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_batch) = batch_number; measurement["batch"] = mx_batch;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial_number;  measurement["trial"] = mx_trial;
	auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = train_number;  measurement["train"] = mx_train;
	auto mx_test = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_test) = test_number;  measurement["test"] = mx_test;
	auto mx_repeat = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_repeat) = repeat_number;  measurement["repeat"] = mx_repeat;

	auto mx_values = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(values.begin(), values.end(), (float *)mxGetData(mx_values)); measurement["values"] = mx_values;

	append(handle->result, { "measurement", "frechet_distance" , "readout" }, measurement);

	update();
}
void Callbacks::callback_measurement_position_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	std::map<std::string, mxArray *> measurement;

	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;

	TRN4CPP::Simulation::decode(simulation_id, frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_batch) = batch_number; measurement["batch"] = mx_batch;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial_number;  measurement["trial"] = mx_trial;
	auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = train_number;  measurement["train"] = mx_train;
	auto mx_test = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_test) = test_number;  measurement["test"] = mx_test;
	auto mx_repeat = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_repeat) = repeat_number;  measurement["repeat"] = mx_repeat;

	auto mx_values = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(values.begin(), values.end(), (float *)mxGetData(mx_values)); measurement["values"] = mx_values;

	append(handle->result, { "measurement", "mean_square_error" , "position" }, measurement);

	update();
}
void Callbacks::callback_measurement_position_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	std::map<std::string, mxArray *> measurement;

	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;

	TRN4CPP::Simulation::decode(simulation_id, frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_batch) = batch_number; measurement["batch"] = mx_batch;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial_number;  measurement["trial"] = mx_trial;
	auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = train_number;  measurement["train"] = mx_train;
	auto mx_test = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_test) = test_number;  measurement["test"] = mx_test;
	auto mx_repeat = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_repeat) = repeat_number;  measurement["repeat"] = mx_repeat;

	auto mx_values = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(values.begin(), values.end(), (float *)mxGetData(mx_values)); measurement["values"] = mx_values;

	append(handle->result, { "measurement", "frechet_distance" , "position" }, measurement);

	update();
}

void Callbacks::callback_performances(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	TRACE_LOGGER;
	std::map<std::string, mxArray *> performances;
	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;

	TRN4CPP::Simulation::decode(simulation_id, frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; performances["condition"] = mx_condition;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_batch) = batch_number; performances["batch"] = mx_batch;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial_number;  performances["trial"] = mx_trial;
	auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = train_number;  performances["train"] = mx_train;
	auto mx_test = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_test) = test_number;  performances["test"] = mx_test;
	auto mx_repeat = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_repeat) = repeat_number;  performances["repeat"] = mx_repeat;

	auto mx_phase = mxCreateString(phase.c_str()); performances["phase"] = mx_phase;
	auto mx_cycles = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_cycles) = cycles_per_second; performances["cycles_per_second"] = mx_cycles;
	auto mx_gflops = mxCreateNumericMatrix(1, 1, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); *(float *)mxGetData(mx_gflops) = gflops_per_second; performances["gflops_per_second"] = mx_gflops;

	append(handle->result, { "recording", "performances"}, performances);
	update();
}
void Callbacks::callback_states(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	std::map<std::string, mxArray *> states;
	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;

	TRN4CPP::Simulation::decode(simulation_id, frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; states["condition"] = mx_condition;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_batch) = batch_number; states["batch"] = mx_batch;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial_number;  states["trial"] = mx_trial;
	auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = train_number;  states["train"] = mx_train;
	auto mx_test = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_test) = test_number;  states["test"] = mx_test;
	auto mx_repeat = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_repeat) = repeat_number;  states["repeat"] = mx_repeat;

	auto mx_phase = mxCreateString(phase.c_str()); states["phase"] = mx_phase;

	auto mx_samples = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(samples.begin(), samples.end(), (float *)mxGetData(mx_samples)); states["samples"] = mx_samples;

	append(handle->result, { "recording", "states", label }, states);
	update();
}
void Callbacks::callback_weights(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch,  const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	std::map<std::string, mxArray *> weights;
	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;

	TRN4CPP::Simulation::decode(simulation_id, frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; weights["condition"] = mx_condition;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_batch) = batch_number; weights["batch"] = mx_batch;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial_number;  weights["trial"] = mx_trial;
	auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = train_number;  weights["train"] = mx_train;

	auto mx_phase = mxCreateString(phase.c_str()); weights["phase"] = mx_phase;

	auto mx_samples = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(samples.begin(), samples.end(), (float *)mxGetData(mx_samples)); weights["samples"] = mx_samples;

	append(handle->result, { "recording", "weights" , label }, weights);
	update();

}
void Callbacks::callback_scheduling(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	std::map<std::string, mxArray *> scheduling;
	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short frontend_number;

	TRN4CPP::Simulation::decode(simulation_id, frontend_number, condition_number, batch_number);

	unsigned short trial_number;
	unsigned short train_number;
	unsigned short test_number;
	unsigned short repeat_number;

	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial_number, train_number, test_number, repeat_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; scheduling["condition"] = mx_condition;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_batch) = batch_number; scheduling["batch"] = mx_batch;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial_number;  scheduling["trial"] = mx_trial;
	auto mx_train = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_train) = train_number;  scheduling["train"] = mx_train;

	auto mx_offsets = mxCreateNumericMatrix(offsets.size(), 1, mxClassID::mxINT32_CLASS, mxComplexity::mxREAL); std::copy(offsets.begin(), offsets.end(), (int *)mxGetData(mx_offsets)); scheduling["offsets"] = mx_offsets;
	auto mx_durations = mxCreateNumericMatrix(durations.size(), 1, mxClassID::mxINT32_CLASS, mxComplexity::mxREAL); std::copy(durations.begin(), durations.end(), (int *)mxGetData(mx_durations)); scheduling["durations"] = mx_durations;

	append(handle->result, { "recording", "scheduling" }, scheduling);
	update();
}


void Callbacks::callback_results(const unsigned short &condition_number, const std::size_t &generation_number, const std::vector<std::pair<std::map<std::string, std::string>, std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>>> &results)
{
	std::size_t configuration_number = 0;
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
	}
}

void Callbacks::callback_solutions(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions)
{
	std::vector<std::string> rows;
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
	}
}