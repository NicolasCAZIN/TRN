#ifdef USE_VLD
#include <vld.h>
#endif 
#include <boost/signals2.hpp>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include <condition_variable>
#include <vector>
#include <random>
#include <mutex>
#include <thread>
#include <algorithm>    // copy
#include <iostream>
#include <fstream>
#include <functional>
#include <string>
#include <list>
#include <queue>
#include "TRN4CPP/TRN4CPP.h"
#include "Helper/Queue.h"

size_t total_simulations = 0;
size_t total_cycles = 0;
float total_gflops = 0.0f;
float total_seconds = 0.0f;


struct Node
{
	std::vector<int> ranks;
	std::string host;
	unsigned int index;
	std::string name;
};


struct Perf
{
	size_t simulations;
	size_t cycles;
	float gflops;
	float seconds;
	float throughput;
	float speed;
};

std::map<std::pair<std::string, unsigned int>, Perf> node_perfs;
std::map<std::string, Perf> host_perfs;
std::map<std::pair<std::string, unsigned int>, Node> nodes;

std::map<int, std::pair<std::string, unsigned int>> processor_node;
std::map<unsigned int, int> simulation_processor;
std::map<std::string, std::vector<int>> host_rank;
std::mutex processor;

static void on_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)
{
	std::unique_lock<std::mutex> lock(processor);
	
	auto key = std::make_pair(host, index);
	if (nodes.find(key) == nodes.end())
	{
		nodes[key].host = host;
		nodes[key].name = name;
		nodes[key].index = index;
	}
	nodes[key].ranks.push_back(rank);
	processor_node[rank] = key;
	host_rank[host].push_back(rank);
}

static void on_allocation(const unsigned int &id, const int &rank)
{
	std::unique_lock<std::mutex> lock(processor);

	std::cout << "id #" << id << " allocated on " << rank << std::endl;

	simulation_processor[id] = rank;
}


static void on_performances(const unsigned int &id, const std::string &phase, const std::size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)
{

	auto throughput = batch_size * cycles / seconds;
	auto speed = batch_size * gflops / seconds;

	std::cout << "id = " << std::to_string(id) <<
		", phase = " << phase <<
		", throughput = " << std::fixed << throughput << " Hz, " <<
		" speed = " << std::fixed << speed << " GFlops/s" << std::endl;

	//if (phase == "TRAIN")
	{
		std::unique_lock<std::mutex> lock(processor);
		total_gflops += batch_size * gflops;
		total_cycles += batch_size * cycles;
		total_seconds += seconds;

		total_simulations += batch_size;


		auto node_key = processor_node[simulation_processor[id]];
		auto node = nodes[node_key];


		node_perfs[node_key].cycles += batch_size * cycles;
		node_perfs[node_key].seconds += seconds;
		node_perfs[node_key].gflops += batch_size * gflops;
		node_perfs[node_key].throughput += throughput;
		node_perfs[node_key].speed += speed;
		node_perfs[node_key].simulations++;
		host_perfs[node.host].cycles += batch_size * cycles;
		host_perfs[node.host].seconds += seconds;
		host_perfs[node.host].gflops += batch_size * gflops;
		host_perfs[node.host].throughput += throughput;
		host_perfs[node.host].speed += speed;
		host_perfs[node.host].simulations++;
	}
}

TRN::Helper::Queue<std::pair<std::string, cv::Mat>> to_display;
std::mutex windows_mutex;
std::map<std::string, cv::Mat> windows;

static void on_states(const unsigned int &id, const std::string &phase, const std::string &label, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)
{
	//cv::namedWindow(label, 1);
	//std::cout << "id = " << std::to_string(id) << ", states = " << label << ", rows = " << rows << ", cols = " << cols << std::endl;
//	cv::Mat my_mat(rows, cols, CV_32FC1, (char *)data.data());
	cv::Mat copy;
	cv::Mat(rows, cols, CV_32FC1, (char *)data.data()).copyTo(copy);
	to_display.enqueue(std::make_pair("id #" + std::to_string(id) + " " + label + "@" + phase, copy));
	//cv::imshow(label, windows[label]);
//	auto mat = windows[label];

}
static void on_weights(const unsigned int &id, const std::string &label, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)
{
	int histSize = 256;
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	cv::Mat histImage(cv::Size(hist_h, hist_w), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat mat(rows, cols, CV_32FC1, (char *)data.data());
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


	//std::cout << "id = " << std::to_string(id) << ", weights = " << label << ", rows = " << rows << ", cols = " << cols << std::endl;
	
	to_display.enqueue(std::make_pair(label, histImage));
	//auto mat = windows[label];



}
std::vector<float> current;
std::map<unsigned int, std::function<void(const unsigned int &id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)>> robot_estimated_position;
std::map<unsigned int, std::function<void(const unsigned int &id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)>> robot_perceived_stimulus;
std::vector<float> place_cells_response;

std::function<float(const float &v)> jitter_x;
std::function<float(const float &v)> jitter_y;
std::function<void(const float &x, const float &y, float *A)> place_cell_activation;


static const float RADIUS_THRESHOLD = 0.2f;
static const float JITTER_ABS = 0.0f;


inline float
BilinearInterpolation(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y)
{
	float x2x1, y2y1, x2x, y2y, yy1, xx1;
	x2x1 = x2 - x1;
	y2y1 = y2 - y1;
	x2x = x2 - x;
	y2y = y2 - y;
	yy1 = y - y1;
	xx1 = x - x1;
	return 1.0f / (x2x1 * y2y1) * (
		q11 * x2x * y2y +
		q21 * xx1 * y2y +
		q12 * x2x * yy1 +
		q22 * xx1 * yy1
		);
}
std::size_t stimulus_size;

void activation_pattern(const std::size_t &pc_rows, const std::size_t &pc_cols, const std::pair<float, float> &X, const std::pair<float, float> &Y, const float &x, const float &y, float *A)
{
	const auto stride = pc_cols * pc_rows;
	const auto x_range = X.second - X.first;
	const auto y_range = Y.second - Y.first;
	const auto x_step = x_range / (pc_cols - 1);
	const auto y_step = y_range / (pc_rows - 1);

	const auto ix1 = (std::size_t)((pc_cols - 1) * ((x - X.first) / x_range));
	const auto iy1 = (std::size_t)((pc_rows - 1) * ((y - Y.first) / y_range));
	const auto ix2 = (ix1 == pc_cols - 1) ? ix1 : ix1 + 1;
	const auto iy2 = (iy1 == pc_rows - 1) ? iy1 : iy1 + 1;
	const auto x1 = ix1 * x_step + X.first;
	const auto x2 = ix2 * x_step + X.first;
	const auto y1 = iy1 * y_step + Y.first;
	const auto y2 = iy2 * y_step + Y.first;

#pragma omp parallel for
	for (int k = 0; k < stimulus_size; k++)
	{
		const auto Q = &place_cells_response[k*stride];
		const auto q11 = Q[iy1 * pc_cols + ix1];
		const auto q12 = Q[iy2 * pc_cols + ix1];
		const auto q21 = Q[iy1 * pc_cols + ix2];
		const auto q22 = Q[iy2 * pc_cols + ix2];
		const auto q = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
		A[k] = q;
	}
}

static void robot_predicted_stimulus(const unsigned int &id, const std::vector<float> &predicted_stimulus, const std::size_t &rows, const std::size_t &cols)
{

}

static void robot_predicted_position(const unsigned int &id, const std::vector<float> &predicted_position, const std::size_t &rows, const std::size_t &cols)
{
	std::vector<float> effective_position(predicted_position.size());
	std::vector<float> perceived_stimulus(rows * stimulus_size);
#pragma omp parallel for
	for (int row = 0; row < rows; row++)
	{
		auto idx_x = row * cols;
		auto idx_y = idx_x + 1;
		effective_position[idx_x] = jitter_x(predicted_position[idx_x]);
		effective_position[idx_y] = jitter_y(predicted_position[idx_y]);
		place_cell_activation(effective_position[idx_x], effective_position[idx_y], &perceived_stimulus[row * stimulus_size]);
	}

	robot_estimated_position[id](id, effective_position, rows, cols);
	robot_perceived_stimulus[id](id, perceived_stimulus, rows, stimulus_size);
}
void initialize_stimulus(const std::pair<float, float> &X, const std::pair<float, float> &Y, const std::vector<float> &place_cells_response, const std::size_t &stimulus_size, const std::size_t &pc_rows, const std::size_t &pc_cols,
	const std::vector<float> &position, const std::size_t &observations, std::vector<float> &activation)
{

	activation.resize(observations * stimulus_size);


	for (std::size_t t = 0; t < observations; t++)
	{
		const auto x = position[t * 2];
		const auto y = position[t * 2 + 1];
		const auto &A = &activation[t * stimulus_size];

		activation_pattern( pc_rows, pc_cols, X,Y, x, y, A);

	
	}
}
#include <string>
struct seq
{
	std::vector<float> incoming;
	std::vector<float> expected;
	std::vector<float> position;
	std::vector<float> reward;
	std::size_t observations;
};

std::map<std::string, seq> sequences;

static void initialize_trajectory(const unsigned int id, const std::string &label, const std::pair<float, float> &x, const std::pair<float, float> &y, const std::size_t &pc_rows, const std::size_t &pc_cols)
{
	if (sequences.find(label) == sequences.end())
	{
		auto filename = label + ".csv";


		std::vector<float> stimulus;
		std::vector<float> incoming;
		std::vector<float> expected;
		std::vector<float> position;
		std::vector<float> reward;
	

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

		std::size_t count = 0;
		if (header.size() < 0)
			throw std::runtime_error("Format error at line " + count);

		while (std::getline(in, line))
		{
			boost::tokenizer<boost::char_separator<char>>  ttok(line, sep);
			tuple.assign(ttok.begin(), ttok.end());
			position.push_back(std::stof(tuple[4]));
			position.push_back(std::stof(tuple[5]));
			reward.push_back(tuple[8] == "true" ? 1.0f : 0.0f);

			count++;
		}
		

		initialize_stimulus(x, y, place_cells_response, stimulus_size, pc_rows, pc_cols, position, count, stimulus);
		count--;

		incoming.resize(stimulus_size * count);
		expected.resize(stimulus_size * count);
		
		position = std::vector<float>(position.begin() + 2, position.end());
		reward = std::vector<float>(reward.begin() + 1, reward.end());


		std::copy(stimulus.begin() + stimulus_size, stimulus.end(), expected.begin());
		std::copy(stimulus.begin(), stimulus.end() - stimulus_size, incoming.begin());

		/*cv::Mat cv_ex(count, stimulus_size, CV_32F, expected.data());
		cv::Mat cv_in(count, stimulus_size, CV_32F, incoming.data());*/
		assert(expected.size() == count * stimulus_size);
		assert(incoming.size() == count * stimulus_size);
		assert(position.size() == count * 2);
		assert(reward.size() == count);
		sequences[label].expected = expected;
		sequences[label].incoming = incoming;
		sequences[label].position = position;
		sequences[label].reward = reward;
		sequences[label].observations = count;
	}
	
	
	TRN4CPP::declare_sequence(id, label, "INC", sequences[label].incoming, sequences[label].observations);
	TRN4CPP::declare_sequence(id, label, "EXP", sequences[label].expected, sequences[label].observations);
	TRN4CPP::declare_sequence(id, label, "POS", sequences[label].position, sequences[label].observations);
	TRN4CPP::declare_sequence(id, label, "REW", sequences[label].reward, sequences[label].observations);
	

}

static void initialize_place_cell_pattern(const unsigned int &id, const std::string &label)
{
	auto filename = label + ".csv";
	std::ifstream in(filename);
	if (!in.is_open())
		throw std::invalid_argument("File " + filename + " can't be opened");

	std::vector< std::string > body;
	std::vector< std::string > header;

	boost::char_separator<char> sep("\t ");

	{
		std::string line;
		if (!std::getline(in, line))
			throw std::runtime_error("Empty header");
		boost::tokenizer<boost::char_separator<char>>  htok(line, sep);
		header.assign(htok.begin(), htok.end());
	}
	auto observations = std::stoi(header[0]);
	auto stimulus_size = std::stoi(header[1]);
	{
		std::string line;
		if (!std::getline(in, line))
			throw std::runtime_error("Empty header");
		boost::tokenizer<boost::char_separator<char>>  htok(line, sep);
		body.assign(htok.begin(), htok.end());
	}
	std::vector<float> stimulus(body.size());

	std::transform(body.begin(), body.end(), stimulus.begin(),
		[](const std::string &arg) { return std::stof(arg); });

	observations--;

	std::vector<float> incoming(stimulus_size * observations);
	std::vector<float> expected(stimulus_size * observations);


	std::copy(stimulus.begin() + stimulus_size, stimulus.end(), expected.begin());
	std::copy(stimulus.begin(), stimulus.end() - stimulus_size, incoming.begin());

	TRN4CPP::declare_sequence(id, label, "INC", incoming, observations);
	TRN4CPP::declare_sequence(id, label, "EXP", expected, observations);
}


static void initialize_place_cells_response(const std::size_t &rows, const std::size_t &cols,
											const std::string &filename, std::vector<float> &response, std::size_t &stimulus_size )
{
	std::vector<float> pc_x;
	std::vector<float> pc_y;
	std::vector<float> pc_radius;

	auto x = std::make_pair(std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
	auto y = std::make_pair(std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

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
	std::size_t count = 0;
	if (header.size() < 0) 
		throw std::runtime_error("Format error at line " + count);

	while (std::getline(in, line))
	{
		boost::tokenizer<boost::char_separator<char>>  ttok(line, sep);
		tuple.assign(ttok.begin(), ttok.end());
		pc_x.push_back(std::stof(tuple[3]));
		pc_y.push_back(std::stof(tuple[4]));
		pc_radius.push_back(std::stof(tuple[5]));
		if (tuple.size() != header.size())
			throw std::runtime_error("Format error at line " + count);
		x.first = std::min(x.first, pc_x[count]);
		x.second = std::max(x.second, pc_x[count]);
		y.first = std::min(y.first, pc_y[count]);
		y.second = std::max(y.second, pc_y[count]);
		count++;
	}
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dist(-JITTER_ABS, JITTER_ABS);
	std::function<float(void)> dice = std::bind(dist, gen);
	auto jitter = [dice](const std::pair<float, float> &b, const float &v)
	{
		auto jittered = v + dice();
		return jittered > b.second ? b.second : jittered < b.first ? b.first : jittered;
	};

	jitter_x = std::bind(jitter, x, std::placeholders::_1);
	jitter_y = std::bind(jitter, y, std::placeholders::_1);

	stimulus_size = pc_x.size();

	response.resize(rows * cols * stimulus_size);
	std::vector<float> g_x(cols);
	std::vector<float> g_y(rows);
	
#pragma omp parallel sections
	{
#pragma omp section
		{
			auto y_range = y.second - y.first;
			for (std::size_t row = 0; row < rows; row++)
				g_y[row] = ((float)row / ((float)(rows - 1))) * y_range + y.first;
		}
#pragma omp section
		{
			auto x_range = x.second - x.first;
			for (std::size_t col = 0; col < cols; col++)
				g_x[col] = ((float)col / ((float)(cols - 1))) * x_range + x.first;
		}
	}
#pragma omp parallel for
	for (int k = 0; k < stimulus_size; k++)
	{
		const auto K = (pc_radius[k] * pc_radius[k]) / logf(RADIUS_THRESHOLD);
		std::vector<float>      dx2(cols);

		for (std::size_t col = 0; col < cols; col++)
		{
			const auto p = pc_x[k] - g_x[col];
			dx2[col] = (p * p) / K;
		}
		for (std::size_t row = 0; row < rows; row++)
		{
			const auto p = pc_y[k] - g_y[row];
			const auto dy2 = (p * p) / K;
			const auto offset = k * (cols*rows);
			for (std::size_t col = 0; col < cols; col++)
			{
				response[offset + row * cols + col] = expf(dx2[col] + dy2);
			}
		}
	}
}
#include <numeric>
static void position_mse(const unsigned int &id, const std::vector<float> &value,  const std::size_t &rows, const std::size_t &cols)
{
	std::cout << id << " position rmse " << sqrtf(std::accumulate(value.begin(), value.end(), 0.0f) / value.size()) << std::endl;
}

static void position_frechet(const unsigned int &id, const std::vector<float> &value, const std::size_t &rows, const std::size_t &cols)
{

}

static void position_custom(const unsigned int &id, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &rows, const std::size_t &cols)
{

}

static void readout_mse(const unsigned int &id, const std::vector<float> &value, const std::size_t &rows, const std::size_t &cols)
{
	std::cout << id << " readout rmse " << sqrtf(std::accumulate(value.begin(), value.end(), 0.0f) / value.size())  << std::endl;

}

static void readout_frechet(const unsigned int &id, const std::vector<float> &value, const std::size_t &rows, const std::size_t &cols)
{

}
static void readout_custom(const unsigned int &id, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &rows, const std::size_t &cols)
{

}

int main(int argc, char *argv[])
{
	try
	{
	
		/*std::thread display([]() {

			//std::cout << "waiting for q stroke" << std::endl;
			//while (cvWaitKey(100) != 'q');
			std::pair<std::string, cv::Mat> p;
			while (to_display.dequeue(p))
			{
				std::unique_lock<std::mutex> lock(windows_mutex);
				windows[p.first] = p.second;
			}
		});

		std::thread expose([]() {

			cv::namedWindow("dummy");
			while (cvWaitKey(100) != 27)
			{
				std::unique_lock<std::mutex> lock(windows_mutex);
				for (auto p : windows)
				{
					cv::namedWindow(p.first, CV_WINDOW_AUTOSIZE);
					cv::imshow(p.first, p.second);
				}
			}
			to_display.invalidate();
		});*/

		/*auto backend = TRN::Remote::Backend::create("127.0.0.1", 12345);
		auto server = TRN::Network::Server::create("127.0.0.1", "12345");
		server->run();

		return 0;*/

		auto pc_rows = 100;
		auto pc_cols = 100;
		auto sigma = 1;


		auto x = std::make_pair(-1.0f, 1.0f);
		auto y = std::make_pair(-1.0f, 1.0f);

		initialize_place_cells_response(pc_rows, pc_cols, "placecells.csv", place_cells_response, stimulus_size);

		place_cell_activation = std::bind(activation_pattern, pc_rows, pc_cols, x, y, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);









		auto seed = 2;
		TRN4CPP::install_processor(on_processor);
		TRN4CPP::install_allocation(on_allocation);
	
		TRN4CPP::initialize_distributed(argc, argv);


		/*TRN4CPP::initialize_local(
		{
			1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4
			
		});*/
		//TRN4CPP::initialize_local();

		auto epochs = 1;
		
		auto reservoir_size = 1024;
		auto prediction_size = stimulus_size;
		auto leak_rate = 0.85f;
		auto learning_rate = 1e-5f;
		auto initial_state_scale = 0.01f;
		auto radius = 0.5f;
		//	auto observations = 130*5;
		auto snippet_size = 10;
		auto snippet_number = 20;
		auto time_budget = epochs * snippet_size * snippet_number;
		auto preamble = 10;
		auto batch_size = 100;

		/*	for (int row = stimulus_size, col = 0; row < stimulus_size*2; row++, col++)
			{
				stimulus[row * stimulus_size + stimulus_size - col-1] = 1.0f;
			}*/
		const size_t ID = 400;
		const size_t TRIALS = 10;
		const size_t WALKS = 0;
		std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
		



//#pragma omp parallel for 
		for (int id = 0; id < ID; id++)
		{
	
				TRN4CPP::allocate(id);

				TRN4CPP::configure_begin(id);
				//TRN4CPP::configure_scheduler_snippets(id,snippet_size, time_budget);
				TRN4CPP::configure_scheduler_tiled(id, epochs);

				TRN4CPP::configure_loop_copy(id, batch_size, stimulus_size);
				//TRN4CPP::configure_loop_custom(id, stimulus_size, robot_action, robot_perception);
				//TRN4CPP::configure_loop_spatial_filter(id, batch_size, stimulus_size, robot_predicted_position, robot_estimated_position[id], robot_predicted_stimulus, robot_perceived_stimulus[id], pc_rows, pc_cols, x, y, place_cells_response, sigma, radius, "POS");
				TRN4CPP::configure_measurement_position_mean_square_error(id, batch_size, position_mse);
				TRN4CPP::configure_measurement_readout_mean_square_error(id, batch_size, readout_mse);
				//TRN4CPP::configure_measurement_position_frechet_distance(id, position_frechet);
				//TRN4CPP::configure_measurement_position_custom(id, position_custom);

				//TRN4CPP::configure_measurement_readout_frechet_distance(id, readout_frechet);
				//TRN4CPP::configure_measurement_readout_custom(id, readout_custom);
				TRN4CPP::configure_reservoir_widrow_hoff(id, stimulus_size, prediction_size, reservoir_size, leak_rate, initial_state_scale, learning_rate, seed, batch_size);
				TRN4CPP::setup_performances(id, on_performances, true, true, true);
				//TRN4CPP::setup_weights(id, on_weights);
				//TRN4CPP::setup_states(id, on_states, true, false, true);



				TRN4CPP::configure_feedforward_uniform(id, -1.0f, 1.0f, 0.0f);
				TRN4CPP::configure_feedback_uniform(id, -1.0f, 1.0f, 0.0f);
				TRN4CPP::configure_recurrent_gaussian(id, 0.0f, 1.0f / sqrtf(reservoir_size));
				TRN4CPP::configure_readout_uniform(id, -1.0e-2f, 1.0e-2f, 0.0f);
				TRN4CPP::configure_end(id);
				//initialize_place_cell_pattern(id, "test");
				initialize_trajectory(id, "abcde", x, y, pc_rows, pc_cols);
				/*initialize_trajectory(id, "ebcda", x, y, pc_rows, pc_cols);
				initialize_trajectory(id, "bacde", x, y, pc_rows, pc_cols);
				initialize_trajectory(id, "abced", x, y, pc_rows, pc_cols);*/

					//const std::vector<std::string> training_sequences = {  "ebcda", "bacde", "abced"};
				const std::vector<std::string> training_sequences = { "abcde" };
				TRN4CPP::declare_set(id, "training", "INC", training_sequences);
				TRN4CPP::declare_set(id, "training", "EXP", training_sequences);
				TRN4CPP::declare_set(id, "training", "POS", training_sequences);
				TRN4CPP::declare_set(id, "training", "REW", training_sequences);

				for (int trial = 0; trial < TRIALS; trial++)
				{
					TRN4CPP::train(id, "training", "INC", "EXP");

					for (int walk = 0; walk < WALKS; walk++)
						TRN4CPP::test(id, "abcde", "INC", "EXP", preamble);
//
				}
				TRN4CPP::deallocate(id);


			//});
		}
		
	

		TRN4CPP::uninitialize();
		


		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> seconds = std::chrono::duration_cast<std::chrono::duration<float>>(t1 - t0);
		std::cout << "simulations / second : " << std::fixed << (total_simulations / seconds.count()) << std::endl;


		
		
		//
		//
	

		for (auto p : nodes)
		{
			auto node_key = p.first;
			auto node = p.second;
			auto throughput = (node_perfs[node_key].throughput)/ node_perfs[node_key].simulations;
			auto speed = (node_perfs[node_key].speed)/ node_perfs[node_key].simulations;
			throughput *= node.ranks.size();
			speed *= node.ranks.size();
			std::cout << node.host << " / device #" << node.index << " ("<< node.name << ")" << " with " << node.ranks.size() << " subscribers : throughput " << throughput << " cycles per second / " << std::fixed << speed << " Gflops/s" << std::endl;
		}
		for (auto p : host_perfs)
		{
			auto throughput = p.second.throughput/ p.second.simulations;
			auto speed = p.second.speed / p.second.simulations;
			throughput *= host_rank[p.first].size();
			speed *= host_rank[p.first].size();
			std::cout << p.first << " : throughput " << throughput << " cycles per second / " << std::fixed << speed << " Gflops/s" << std::endl;
		}
	

		/*if (expose.joinable())
			expose.join();
		if (display.joinable())
			display.join();*/

		return 0;
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}
}