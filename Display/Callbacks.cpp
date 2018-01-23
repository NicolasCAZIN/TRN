#include "stdafx.h"
#include "Callbacks.h"
#include "Helper/Queue.h"
#include "Helper/Logger.h"

static const std::string FILENAME_TOKEN = "FILENAME";
static const std::string WIDTH_TOKEN = "WIDTH";
static const std::string HEIGHT_TOKEN = "HEIGHT";
static const std::string THICKNESS_TOKEN = "THICKNESS";

struct ID
{
	unsigned int batch_number;
	unsigned short condition_number;
	unsigned short number;

	ID(const unsigned long long &simulation_id)
	{
		TRN4CPP::Simulation::decode(simulation_id,number, condition_number, batch_number);
	}
};


struct Callbacks::Handle
{
	std::size_t width;
	std::size_t height;
	std::size_t thickness;
	std::pair<float, float> x;
	std::pair<float, float> y;
	std::thread display;
	std::thread expose;
	TRN::Helper::Queue<std::pair<std::string, cv::Mat>> to_display;
	std::mutex windows_mutex;
	std::mutex traj_mutex;
	std::map<std::string, cv::Mat> windows;
	std::set<std::string> allocated;
	std::map<std::size_t, std::map<std::size_t, std::map<std::size_t, cv::Mat>>> accumulator;
	std::map<std::size_t, std::map<std::size_t, std::map<std::size_t, cv::Mat>>> overall;
	std::map<std::size_t, std::map<std::size_t, std::map<std::size_t, std::size_t>>> trajectories;

	bool expose_required;
};

static std::ostream &operator << (std::ostream &stream, const ID id)
{
	stream << "frontend_number = " << id.number << ", condition_number = " << id.condition_number << ", bundle_size = " << id.batch_number;
	return stream;
}

void Callbacks::initialize(const std::map<std::string, std::string> &arguments)
{
	TRACE_LOGGER;
	if (arguments.find(FILENAME_TOKEN) == arguments.end())
		throw std::invalid_argument("Can't find argument " + FILENAME_TOKEN);
	if (arguments.find(WIDTH_TOKEN) == arguments.end())
		throw std::invalid_argument("Can't find argument " + WIDTH_TOKEN);
	if (arguments.find(HEIGHT_TOKEN) == arguments.end())
		throw std::invalid_argument("Can't find argument " + HEIGHT_TOKEN);
	if (arguments.find(THICKNESS_TOKEN) == arguments.end())
		throw std::invalid_argument("Can't find argument " + THICKNESS_TOKEN);
	auto filename = arguments.at(FILENAME_TOKEN);

	handle = std::make_unique<Handle>();

	handle->width = std::stoi(arguments.at(WIDTH_TOKEN));
	handle->height = std::stoi(arguments.at(HEIGHT_TOKEN));
	handle->thickness = std::stoi(arguments.at(THICKNESS_TOKEN));
	handle->x = std::make_pair(std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
	handle->y = std::make_pair(std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

	std::ifstream in(filename);
	if (!in.is_open())
		throw std::invalid_argument("File " + filename + " can't be opened");

	std::vector< std::string > tuple;
	std::vector< std::string > header;
	std::string line;
	boost::char_separator<char> sep("\t ");

	if (!std::getline(in, line))
		throw std::runtime_error("Empty file");
	boost::tokenizer<boost::char_separator<char>>  htok(line, sep);
	header.assign(htok.begin(), htok.end());

	if (header.size() < 0)
		throw std::runtime_error("Format error at line " + 0);
	int line_number = 1;
	while (std::getline(in, line))
	{
		boost::tokenizer<boost::char_separator<char>>  ttok(line, sep);
		tuple.assign(ttok.begin(), ttok.end());
		auto x_center = std::stof(tuple[3]);
		auto y_center = std::stof(tuple[4]);


		if (tuple.size() != header.size())
			throw std::runtime_error("Format error at line " + line_number);
		handle->x.first = std::min(handle->x.first, x_center);
		handle->x.second = std::max(handle->x.second, x_center);
		handle->y.first = std::min(handle->y.first, y_center);
		handle->y.second = std::max(handle->y.second, y_center);
		line_number++;
	}



	handle->display = std::thread([&]()
	{
		std::pair<std::string, cv::Mat> p;
		while (handle->to_display.dequeue(p))
		{
			std::unique_lock<std::mutex> lock(handle->windows_mutex);
			handle->windows[p.first] = p.second;
		}
		std::unique_lock<std::mutex> lock(handle->windows_mutex);
		handle->expose_required = false;

		INFORMATION_LOGGER << "MONITOR display thread exited";
	});

	handle->expose_required = true;
	handle->expose = std::thread([&]()
	{
		bool stop = false;
		while (!stop)
		{
			auto key = cv::waitKey(100);
			std::unique_lock<std::mutex> lock(handle->windows_mutex);

			//cv::destroyAllWindows();
			for (auto p : handle->windows)
			{
				if (handle->allocated.find(p.first) == handle->allocated.end())
				{
					cv::namedWindow(p.first, CV_WINDOW_AUTOSIZE);
					handle->allocated.insert(p.first);
				}
				cv::imshow(p.first, p.second);
			}
			if (handle->expose_required == false)
				stop = true;
		}
		INFORMATION_LOGGER << "MONITOR expose thread exited";
	});
}

void Callbacks::uninitialize()
{
	TRACE_LOGGER;
	handle->to_display.invalidate();
	if (handle->display.joinable())
	{
		handle->display.join();
	}
	if (handle->expose.joinable())
	{
		handle->expose.join();
	}
	handle.reset();
}

void Callbacks::callback_measurement_readout_raw(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
}

int Callbacks::x_to_col(float x)
{
	return ((((x)-handle->x.first) / (handle->x.second - handle->x.first)) * (handle->width - 1));
}
int Callbacks::y_to_row(float y)
{
	return (handle->height - 1 - (((y)-handle->y.first) / (handle->y.second - handle->y.first)) * (handle->height - 1));
}
void Callbacks::callback_measurement_position_raw(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	unsigned short trial, train, test, repeat;
	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial, train, test, repeat);

	cv::Mat cv_expected(handle->height, handle->width, CV_8UC3);
	cv::Mat cv_predicted(handle->height, handle->width, CV_32F);

	cv_expected = 0.0f;
	cv_predicted = 0.0f;
	float x0, y0, x1, y1;
	for (std::size_t row = 1; row < preamble; row++)
	{
		x0 = primed[(row - 1)* cols + 0];
		y0 = primed[(row - 1)* cols + 1];
		x1 = primed[row * cols + 0];
		y1 = primed[row * cols + 1];

		cv::Point2d pt0(x_to_col(x0), y_to_row(y0));
		cv::Point2d pt1(x_to_col(x1), y_to_row(y1));
		cv::line(cv_expected, pt0, pt1, cv::Scalar(100, 255, 100), handle->thickness, cv::LineTypes::LINE_AA, 0);
	}
	cv::Point2d pt0(x_to_col(x1), y_to_row(y1));
	x0 = expected[0];
	y0 = expected[1];
	cv::Point2d pt1(x_to_col(x0), y_to_row(y0));
	cv::line(cv_expected, pt0, pt1, cv::Scalar(255, 100, 100), handle->thickness, cv::LineTypes::LINE_AA, 0);

	for (std::size_t row = 1; row < rows; row++)
	{
		x0 = expected[(row - 1)* cols + 0];
		y0 = expected[(row - 1)* cols + 1];
		x1 = expected[row * cols + 0];
		y1 = expected[row * cols + 1];

		cv::Point2d pt0(x_to_col(x0), y_to_row(y0));
		cv::Point2d pt1(x_to_col(x1), y_to_row(y1));
		cv::line(cv_expected, pt0, pt1, cv::Scalar(255, 100, 100), handle->thickness, cv::LineTypes::LINE_AA, 0);
	}
	handle->to_display.enqueue(std::make_pair("expected", cv_expected));


	for (std::size_t page = 0; page < pages; page++)
	{
		cv::Mat cv_temp(handle->height, handle->width, CV_32F);
		cv_temp = 0.0f;


		for (std::size_t row = 1; row < rows; row++)
		{
			x0 = predicted[page * rows * cols + (row - 1)* cols + 0];
			y0 = predicted[page * rows * cols + (row - 1)* cols + 1];
			x1 = predicted[page * rows * cols + row * cols + 0];
			y1 = predicted[page * rows * cols + row * cols + 1];

			cv::Point2d pt0(x_to_col(x0), y_to_row(y0));
			cv::Point2d pt1(x_to_col(x1), y_to_row(y1));
			cv::line(cv_temp, pt0, pt1, cv::Scalar(1.0f), handle->thickness, cv::LineTypes::LINE_AA, 0);
		}

		cv_predicted += cv_temp;
	}
	cv_predicted /= pages;

	//to_display.enqueue(std::make_pair("predcited, trial #" + std::to_string(trial), cv_predicted));

	{
		std::unique_lock<std::mutex> lock(handle->traj_mutex);

		if (handle->accumulator[trial][train][test].empty())
		{
			handle->accumulator[trial][train][test] = cv::Mat(handle->height, handle->width, CV_32F);
			handle->accumulator[trial][train][test] = 0.0f;
			handle->trajectories[trial][train][test] = 0;
		}

		handle->accumulator[trial][train][test] += cv_predicted;
		handle->trajectories[trial][train][test]++;
		handle->overall[trial][train][test] = handle->accumulator[trial][train][test] / handle->trajectories[trial][train][test];
		//cv::equalizeHist(cv_accumulator, cv_overall);
		handle->to_display.enqueue(std::make_pair("accumulated, condition #" + std::to_string(ID(simulation_id).condition_number) + ", trial #" + std::to_string(trial)+ ", train #" + std::to_string(train)+ ", test #" + std::to_string(test), handle->overall[train][trial][test]));
	}
}
void Callbacks::callback_measurement_readout_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
}
void Callbacks::callback_measurement_readout_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
}
void Callbacks::callback_measurement_position_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
}
void Callbacks::callback_measurement_position_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
}
void Callbacks::callback_performances(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	TRACE_LOGGER;
}
void Callbacks::callback_states(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	cv::Mat copy;
	cv::Mat(rows, cols, CV_32FC1, (char *)samples.data()).copyTo(copy);
	unsigned short trial, train, test, repeat;
	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial, train, test, repeat);
	handle->to_display.enqueue(std::make_pair("accumulated, condition #" + std::to_string(ID(simulation_id).condition_number) + ", trial #" + std::to_string(trial) + ", train #" + std::to_string(train) + ", test #" + std::to_string(test), copy));
}
void Callbacks::callback_weights(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch,const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	unsigned short trial, train, test, repeat;
	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial, train, test, repeat);
	int histSize = 256;
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	cv::Mat histImage(cv::Size(hist_h, hist_w), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat mat(rows, cols, CV_32FC1, (char *)samples.data());
	cv::Mat hist;

	float range[] = { -1.0f,  1.0f };
	const float* histRange = { range };
	calcHist(&mat, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());



	for (int i = 1; i < histSize; i++)
	{
		cv::line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
	}


	INFORMATION_LOGGER << "id = " << std::to_string(simulation_id) << ", weights = " << label << ", rows = " << rows << ", cols = " << cols;

	handle->to_display.enqueue(std::make_pair("id #" + std::to_string(simulation_id) + " " + label + "@" + phase + " batch #" + std::to_string(batch) + " trial #" + std::to_string(trial), histImage));


}
void Callbacks::callback_scheduling(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	unsigned short trial, train, test, repeat;
	TRN4CPP::Simulation::Evaluation::decode(evaluation_id, trial, train, test, repeat);
	auto min_max = std::minmax_element(offsets.begin(), offsets.end());
	auto ST = std::max(std::abs(*min_max.first), std::abs(*min_max.second)) + 1;
	auto T = offsets.size();
	cv::Mat chronogram(T, ST, CV_32F);
	chronogram = 0.0f;

	for (std::size_t t = 0; t < T; t++)
	{
		auto st = offsets[t];
		if (st > 0)
			chronogram.at<float>(t, st) = 1.0f;
	}

	handle->to_display.enqueue(std::make_pair("scheduling for simulation #" + std::to_string(simulation_id) + ", trial #" + std::to_string(trial)+", train #" + std::to_string(train), chronogram));
}



void Callbacks::callback_results(const unsigned short &condition_number, const std::size_t &generation_number, const std::vector<std::pair<std::map<std::string, std::string>, std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>>> &results)
{
}


void Callbacks::callback_solutions(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions)
{
}