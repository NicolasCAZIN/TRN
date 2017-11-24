#include "stdafx.h"
#include "Callbacks.h"
#include "Helper/Queue.h"

static const std::string DEFAULT_MODE = "w7.3";

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
	mxArray *result;
	std::thread dump;
	TRN::Helper::Queue<std::pair<std::size_t, mxArray *>> to_save;
};

void Callbacks::initialize(const std::map<std::string, std::string> &arguments)
{
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
	if (version > handle->version_saved)
	{
		try
		{
			boost::filesystem::path path(handle->filename);

			auto basename = boost::filesystem::basename(path);
			auto extension = boost::filesystem::extension(path);
			auto parent_directory = path.parent_path();

			auto filename = basename + "_" + std::to_string(version) + extension;
			auto absolute_filename = (parent_directory / filename).string();
			std::cout << "saving to file " << absolute_filename << std::endl;

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
			std::cerr << e.what() << std::endl;
		}
	}
}

void Callbacks::update()
{
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
				auto to_save = handle->result;
				handle->result = mxCreateStructMatrix(1, 1, 0, NULL);
				handle->to_save.enqueue(std::make_pair(handle->version_updated, to_save));
			}
		}
		else
		{
			std::cerr << "file " << handle->filename << " is still being saved. Waiting for the next timeout" << std::endl;
		}
	
		handle->timestamp = std::clock();
	}
}

static void append(mxArray *root, const std::vector<std::string> &path, const std::map<std::string, mxArray *> fields)
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

void Callbacks::callback_measurement_readout_raw(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	std::size_t predicted_size[3] = { cols, rows, pages };
	std::size_t primed_size[2] = { cols, preamble };
	std::size_t expected_size[2] = { cols, rows };
	std::map<std::string, mxArray *> measurement;

	unsigned int simulation_number;
	unsigned short condition_number;
	unsigned short number;
	
	TRN4CPP::Simulation::decode(id, number, condition_number, simulation_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_simulation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_simulation) = simulation_number; measurement["simulation"] = mx_simulation;

	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial;  measurement["trial"] = mx_trial;
	auto mx_evaluation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_evaluation) = evaluation;  measurement["evaluation"] = mx_evaluation;
	auto mx_primed = mxCreateNumericArray(2, primed_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(primed.begin(), primed.end(), (float *)mxGetData(mx_primed)); measurement["primed"] = mx_primed;
	auto mx_predicted = mxCreateNumericArray(3, predicted_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(predicted.begin(), predicted.end(), (float *)mxGetData(mx_predicted)); measurement["predicted"] = mx_predicted;
	auto mx_expected = mxCreateNumericArray(2, expected_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(expected.begin(), expected.end(), (float *)mxGetData(mx_expected)); measurement["expected"] = mx_expected;
	
	append(handle->result, { "measurement", "raw" , "readout"}, measurement);

	update();
}
void Callbacks::callback_measurement_position_raw(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) 
{
	std::size_t predicted_size[3] = { cols, rows, pages };
	std::size_t primed_size[2] = { cols, preamble };
	std::size_t expected_size[2] = { cols, rows };
	std::map<std::string, mxArray *> measurement;

	unsigned int simulation_number;
	unsigned short condition_number;
	unsigned short number;

	TRN4CPP::Simulation::decode(id, number, condition_number, simulation_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_simulation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_simulation) = simulation_number; measurement["simulation"] = mx_simulation;

	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial;  measurement["trial"] = mx_trial;
	auto mx_evaluation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_evaluation) = evaluation;  measurement["evaluation"] = mx_evaluation;
	auto mx_primed = mxCreateNumericArray(2, primed_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(primed.begin(), primed.end(), (float *)mxGetData(mx_primed)); measurement["primed"] = mx_primed;
	auto mx_predicted = mxCreateNumericArray(3, predicted_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(predicted.begin(), predicted.end(), (float *)mxGetData(mx_predicted)); measurement["predicted"] = mx_predicted;
	auto mx_expected = mxCreateNumericArray(2, expected_size, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(expected.begin(), expected.end(), (float *)mxGetData(mx_expected)); measurement["expected"] = mx_expected;

	append(handle->result, { "measurement", "raw" , "position" }, measurement);

	update();
}
void Callbacks::callback_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	std::map<std::string, mxArray *> measurement;

	unsigned int simulation_number;
	unsigned short condition_number;
	unsigned short number;

	TRN4CPP::Simulation::decode(id, number, condition_number, simulation_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_simulation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_simulation) = simulation_number; measurement["simulation"] = mx_simulation;

	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial;  measurement["trial"] = mx_trial;
	auto mx_evaluation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_evaluation) = evaluation;  measurement["evaluation"] = mx_evaluation;
	auto mx_values = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(values.begin(), values.end(), (float *)mxGetData(mx_values)); measurement["values"] = mx_values;

	append(handle->result, { "measurement", "mean_square_error" , "readout" }, measurement);

	update();
}
void Callbacks::callback_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	std::map<std::string, mxArray *> measurement;

	unsigned int simulation_number;
	unsigned short condition_number;
	unsigned short number;

	TRN4CPP::Simulation::decode(id, number, condition_number, simulation_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_simulation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_simulation) = simulation_number; measurement["simulation"] = mx_simulation;

	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial;  measurement["trial"] = mx_trial;
	auto mx_evaluation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_evaluation) = evaluation;  measurement["evaluation"] = mx_evaluation;
	auto mx_values = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(values.begin(), values.end(), (float *)mxGetData(mx_values)); measurement["values"] = mx_values;

	append(handle->result, { "measurement", "frechet_distance" , "readout" }, measurement);

	update();
}
void Callbacks::callback_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	std::map<std::string, mxArray *> measurement;

	unsigned int simulation_number;
	unsigned short condition_number;
	unsigned short number;

	TRN4CPP::Simulation::decode(id, number, condition_number, simulation_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_simulation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_simulation) = simulation_number; measurement["simulation"] = mx_simulation;

	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial;  measurement["trial"] = mx_trial;
	auto mx_evaluation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_evaluation) = evaluation;  measurement["evaluation"] = mx_evaluation;
	auto mx_values = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(values.begin(), values.end(), (float *)mxGetData(mx_values)); measurement["values"] = mx_values;

	append(handle->result, { "measurement", "mean_square_error" , "position" }, measurement);

	update();
}
void Callbacks::callback_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	std::map<std::string, mxArray *> measurement;

	unsigned int simulation_number;
	unsigned short condition_number;
	unsigned short number;

	TRN4CPP::Simulation::decode(id, number, condition_number, simulation_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; measurement["condition"] = mx_condition;
	auto mx_simulation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_simulation) = simulation_number; measurement["simulation"] = mx_simulation;

	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial;  measurement["trial"] = mx_trial;
	auto mx_evaluation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_evaluation) = evaluation;  measurement["evaluation"] = mx_evaluation;
	auto mx_values = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(values.begin(), values.end(), (float *)mxGetData(mx_values)); measurement["values"] = mx_values;

	append(handle->result, { "measurement", "frechet_distance" , "position" }, measurement);

	update();
}

void Callbacks::callback_performances(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	std::map<std::string, mxArray *> performances;
	unsigned int simulation_number;
	unsigned short condition_number;
	unsigned short number;

	TRN4CPP::Simulation::decode(id, number, condition_number, simulation_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; performances["condition"] = mx_condition;
	auto mx_simulation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_simulation) = simulation_number; performances["simulation"] = mx_simulation;

	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial;  performances["trial"] = mx_trial;
	auto mx_evaluation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_evaluation) = evaluation;  performances["evaluation"] = mx_evaluation;
	auto mx_phase = mxCreateString(phase.c_str()); performances["phase"] = mx_phase;
	auto mx_cycles = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_cycles) = cycles_per_second; performances["cycles_per_second"] = mx_cycles;
	auto mx_gflops = mxCreateNumericMatrix(1, 1, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); *(float *)mxGetData(mx_gflops) = gflops_per_second; performances["gflops_per_second"] = mx_gflops;

	append(handle->result, { "recording", "performances"}, performances);
	update();
}
void Callbacks::callback_states(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	std::map<std::string, mxArray *> states;
	unsigned int simulation_number;
	unsigned short condition_number;
	unsigned short number;

	TRN4CPP::Simulation::decode(id, number, condition_number, simulation_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; states["condition"] = mx_condition;
	auto mx_simulation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_simulation) = simulation_number; states["simulation"] = mx_simulation;

	auto mx_phase = mxCreateString(phase.c_str()); states["phase"] = mx_phase;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial; states["trial"] = mx_trial;
	auto mx_evaluation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_evaluation) = evaluation; states["evaluation"] = mx_evaluation;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_batch) = batch; states["batch"] = mx_batch;

	auto mx_samples = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(samples.begin(), samples.end(), (float *)mxGetData(mx_samples)); states["samples"] = mx_samples;

	append(handle->result, { "recording", "states" }, states);
	update();
}
void Callbacks::callback_weights(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	std::map<std::string, mxArray *> weights;
	unsigned int simulation_number;
	unsigned short condition_number;
	unsigned short number;

	TRN4CPP::Simulation::decode(id, number, condition_number, simulation_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; weights["condition"] = mx_condition;
	auto mx_simulation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_simulation) = simulation_number; weights["simulation"] = mx_simulation;

	auto mx_phase = mxCreateString(phase.c_str()); weights["phase"] = mx_phase;
	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial; weights["trial"] = mx_trial;
	auto mx_batch = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_batch) = batch; weights["batch"] = mx_batch;

	auto mx_samples = mxCreateNumericMatrix(cols, rows, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL); std::copy(samples.begin(), samples.end(), (float *)mxGetData(mx_samples)); weights["samples"] = mx_samples;

	append(handle->result, { "recording", "weights" , phase, label }, weights);
	update();

}
void Callbacks::callback_scheduling(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	std::map<std::string, mxArray *> scheduling;
	unsigned int simulation_number;
	unsigned short condition_number;
	unsigned short number;

	TRN4CPP::Simulation::decode(id, number, condition_number, simulation_number);

	auto mx_condition = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT16_CLASS, mxComplexity::mxREAL); *(unsigned short *)mxGetData(mx_condition) = condition_number; scheduling["condition"] = mx_condition;
	auto mx_simulation = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL); *(unsigned int *)mxGetData(mx_simulation) = simulation_number; scheduling["simulation"] = mx_simulation;

	auto mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL); *(unsigned long long *)mxGetData(mx_trial) = trial; scheduling["trial"] = mx_trial;
	auto mx_offsets = mxCreateNumericMatrix(offsets.size(), 1, mxClassID::mxINT32_CLASS, mxComplexity::mxREAL); std::copy(offsets.begin(), offsets.end(), (int *)mxGetData(mx_offsets)); scheduling["offets"] = mx_offsets;
	auto mx_durations = mxCreateNumericMatrix(durations.size(), 1, mxClassID::mxINT32_CLASS, mxComplexity::mxREAL); std::copy(durations.begin(), durations.end(), (int *)mxGetData(mx_durations)); scheduling["durations"] = mx_durations;

	append(handle->result, { "recording", "scheduling" }, scheduling);
	update();
}

