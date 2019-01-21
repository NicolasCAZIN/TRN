#include "stdafx.h"
#include "TRN4CPP\Basic.h"
#include "TRN4CPP\Simplified.h"
#include "TRN4CPP\Sequences.h"

template <typename Input, typename Output>
static Output transform(const  void *data, const std::size_t &size)
{
	Output output(size);

	std::transform((const Input *)data, (const Input *)data + size, output.begin(), [](const Input &x) { return (Output::value_type)x; });

	return output;
}
template <typename Output>
static Output transform(const mxArray *array)
{
	auto size = mxGetNumberOfElements(array);
	auto data = mxGetData(array);
	if (!mxIsNumeric(array))
	{
		mexErrMsgTxt("argument is not a numeric array");
	}
	Output output;
	switch (mxGetClassID(array))
	{
		case mxClassID::mxSINGLE_CLASS:
			output = transform<float, Output>(data, size);
			break;
		case mxClassID::mxDOUBLE_CLASS:
			output = transform<double, Output>(data, size);
			break;
		case mxClassID::mxINT8_CLASS:
			output = transform<char, Output>(data, size);
			break;
		case mxClassID::mxINT16_CLASS:
			output = transform<short, Output>(data, size);
			break;
		case mxClassID::mxINT32_CLASS:
			output = transform<int, Output>(data, size);
			break;
		case mxClassID::mxINT64_CLASS:
			output = transform<long long, Output>(data, size);
			break;
		case mxClassID::mxUINT8_CLASS:
			output = transform<unsigned char, Output>(data, size);
			break;
		case mxClassID::mxUINT16_CLASS:
			output = transform<unsigned short, Output>(data, size);
			break;
		case mxClassID::mxUINT32_CLASS:
			output = transform<unsigned int, Output>(data, size);
			break;
		case mxClassID::mxUINT64_CLASS:
			output = transform<unsigned long long, Output>(data, size);
			break;
		default:
			mexErrMsgTxt("unsupported numeric format");
			break;
	}

	return output;
}

static std::list<mxArray *> to_free;
static void free_temporary()
{
	for (auto ptr : to_free)
	{
		mxDestroyArray(ptr);
	}
	to_free.clear();
}
static std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> scheduler_request;
static std::function<void(const unsigned long long &simulation_id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> scheduler_reply;

static void uninitialize()
{
	TRN4CPP::Engine::uninitialize();
	free_temporary();
	scheduler_reply = NULL;
	
}

namespace mexplus 
{
	template<>
	void MxArray::to(const mxArray* array, std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> * value)
	{
		if (!value)
		{
			mexErrMsgTxt("Null pointer exception.");
		}

		if (!mxIsFunctionHandle(array))
		{
			mexErrMsgTxt("argument is not a function handle");
		}


		mxArray *functor = mxDuplicateArray(array);
		if (!mxIsFunctionHandle(functor))
		{
			mexErrMsgTxt("argument is not a function handle");
		}
		mexMakeArrayPersistent(functor);
		to_free.push_back(functor);

		*value = [functor](const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
		{
			mxArray *mx_id = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT32_CLASS, mxComplexity::mxREAL);
			mxArray *mx_seed = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL);
			mxArray *mx_trial = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL);
			mxArray *mx_elements = mxCreateNumericMatrix(elements.size(), 1, mxClassID::mxSINGLE_CLASS, mxComplexity::mxREAL);
			mxArray *mx_rows = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL);
			mxArray *mx_cols = mxCreateNumericMatrix(1, 1, mxClassID::mxUINT64_CLASS, mxComplexity::mxREAL);
			mxArray *mx_offsets = mxCreateNumericMatrix(offsets.size(), 1, mxClassID::mxINT32_CLASS, mxComplexity::mxREAL);
			mxArray *mx_durations = mxCreateNumericMatrix(durations.size(), 1, mxClassID::mxINT32_CLASS, mxComplexity::mxREAL);

			assert(mx_id);
			assert(mx_seed);
			assert(mx_trial);
			assert(mx_elements);
			assert(mx_rows);
			assert(mx_cols);
			assert(mx_offsets);
			assert(mx_durations);

			*(unsigned int *)mxGetData(mx_id) = simulation_id;
			*(unsigned long long *)mxGetData(mx_seed) = seed;
			*(unsigned long long *)mxGetData(mx_trial) = trial;
			std::copy(elements.begin(), elements.end(), (float *)mxGetData(mx_elements));
			*(unsigned long long *)mxGetData(mx_rows) = rows;
			*(unsigned long long *)mxGetData(mx_cols) = cols;
			std::copy(offsets.begin(), offsets.end(), (int *)mxGetData(mx_offsets));
			std::copy(durations.begin(), durations.end(), (int *)mxGetData(mx_durations));

	
			mxArray *functor_copy = mxDuplicateArray(functor);

			int nrhs = 9;
			mxArray* prhs[] = { functor_copy, mx_id, mx_seed, mx_trial, mx_elements, mx_rows, mx_cols, mx_offsets, mx_durations };
			mexCallMATLABWithObject(0, NULL, nrhs, prhs, "feval");

			mxDestroyArray(mx_id);
			mxDestroyArray(mx_seed);
			mxDestroyArray(mx_trial);
			mxDestroyArray(mx_elements);
			mxDestroyArray(mx_rows);
			mxDestroyArray(mx_cols);
			mxDestroyArray(mx_offsets);
			mxDestroyArray(mx_durations);
		};
	}
	template<>
	void MxArray::to(const mxArray* array, std::vector<unsigned int> * value)
	{
		if (!value)
		{
			mexErrMsgTxt("Null pointer exception.");
		}

		*value = transform<std::vector<unsigned int>>(array);
	}
	template<>
	void MxArray::to(const mxArray* array, std::vector<float> * value)
	{
		if (!value)
		{
			mexErrMsgTxt("Null pointer exception.");
		}

		*value = transform<std::vector<float>>(array);
	}
}

static const bool BLOCKING = false;


namespace 
{
	/*
	 * Basic API
	 */

	MEX_DEFINE(engine_initialize) (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
	{
		try
		{
			mexplus::InputArguments input(nrhs, prhs, 1);
			mexplus::OutputArguments output(nlhs, plhs, 0);

			auto indices = input.get<std::vector<unsigned int>>(0);

			TRN4CPP::Engine::Backend::Local::initialize(indices);
			mexAtExit(&uninitialize);
		}
		catch (std::exception &e)
		{
			mexErrMsgTxt(e.what());
			TRN4CPP::Engine::uninitialize();
		}
	}

	MEX_DEFINE(engine_uninitialize) (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
	{
		try
		{
			TRN4CPP::Engine::uninitialize();
		}
		catch (std::exception &e)
		{
			mexErrMsgTxt(e.what());
			TRN4CPP::Engine::uninitialize();
		}
	}

	MEX_DEFINE(engine_backend_local_initialize) (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
	{
		try
		{
			mexplus::InputArguments input(nrhs, prhs, 1);
			mexplus::OutputArguments output(nlhs, plhs, 0);

			auto indices = input.get<std::vector<unsigned int>>(0);

			TRN4CPP::Engine::Backend::Local::initialize(indices);
			mexAtExit(&uninitialize);
		}
		catch (std::exception &e)
		{
			mexErrMsgTxt(e.what());
			TRN4CPP::Engine::uninitialize();
		}
	}

	MEX_DEFINE(engine_backend_initialize_remote) (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
	{
		try
		{
			mexplus::InputArguments input(nrhs, prhs, 2);
			mexplus::OutputArguments output(nlhs, plhs, 0);

			auto host = input.get<std::string>(0);
			auto port = input.get<unsigned short>(1);

			TRN4CPP::Engine::Backend::Remote::initialize(host, port);
		}
		catch (std::exception &e)
		{
			mexErrMsgTxt(e.what());
			TRN4CPP::Engine::uninitialize();
		}
		mexAtExit(&uninitialize);
	}

	MEX_DEFINE(simulation_encode) (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
	{
		try
		{
			mexplus::InputArguments input(nrhs, prhs, 4, 1);
			mexplus::OutputArguments output(nlhs, plhs, 1);

			auto label = input.get<std::string>(0);
			auto elements = input.get<std::vector<float>>(1);
			auto rows = input.get<std::size_t>(2);
			auto cols = input.get<std::size_t>(3);
			auto tag = input.get<std::string>("tag", "");
			TRN4CPP::Sequences::declare(label, tag, elements, rows, cols);
		}
		catch (std::exception &e)
		{
			mexErrMsgTxt(e.what());
			TRN4CPP::Engine::uninitialize();
		}
	}


	/*
	 *	Sequences API
	 */
	MEX_DEFINE(sequences_declare) (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
	{
		try
		{
			mexplus::InputArguments input(nrhs, prhs, 4, 1);
			mexplus::OutputArguments output(nlhs, plhs, 0);

			auto label = input.get<std::string>(0);
			auto elements = input.get<std::vector<float>>(1);
			auto rows = input.get<std::size_t>(2);
			auto cols = input.get<std::size_t>(3);
			auto tag = input.get<std::string>("tag", "");
			TRN4CPP::Sequences::declare(label,tag,  elements, rows, cols);
		}
		catch (std::exception &e)
		{
			mexErrMsgTxt(e.what());
			TRN4CPP::Engine::uninitialize();
		}
	}
	/*
	 *	Simplified API
	 */
	MEX_DEFINE(simulation_compute) (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
	{
		try
		{
			mexplus::InputArguments input(nrhs, prhs, 1);
			mexplus::OutputArguments output(nlhs, plhs, 0);

			auto scenario_filename = input.get<std::string>(0);
			TRN4CPP::Simulation::compute(scenario_filename);
		}
		catch (std::exception &e)
		{
			mexErrMsgTxt(e.what());
			TRN4CPP::Engine::uninitialize();
		}
	}
};

MEX_DISPATCH