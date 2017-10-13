#include "stdafx.h"
#include "Simplified.h"
#include "Engine/Frontend.h"
#include "Callbacks.h"
#include "Custom.h"

extern std::shared_ptr<TRN::Engine::Frontend> frontend;

extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_raw;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_raw;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> on_mutator;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduler;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedforward;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedback;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_readout;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> on_recurrent;

extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_mean_square_error;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_frechet_distance;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_mean_square_error;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_frechet_distance;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> on_performances;
extern std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_states;
extern std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_weights;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduling;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> on_position;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> on_stimulus;

static const std::string experiment_name = "EXPERIMENT";
static const std::string declaration_name = "DECLARATION";
static const std::string configuration_name = "CONFIGURATION";
static const std::string execution_name = "EXECUTION";
static const std::string set_name = "SET";
static const std::string condition_name = "CONDITION";
static const std::string loop_name = "LOOP";
static const std::string scheduler_name = "SCHEDULER";
static const std::string reservoir_name = "RESERVOIR";
static const std::string label_name = "LABEL";
static const std::string tag_name = "TAG";
static const std::string sequence_name = "SEQUENCE";
static const std::string measurement_name = "MEASUREMENT";
static const std::string recording_name = "RECORDING";
static const std::string model_name = "MODEL";
static const std::string data_name = "DATA";
static const std::string result_name = "RESULT";
static const std::string test_name = "TEST";
static const std::string train_name = "TRAIN";
static const std::string trial_name = "TRIAL";
static const std::string mutator_name = "MUTATOR";
static const std::string decorator_name = "DECORATOR";
static const std::string variable_name = "VARIABLE";
static const std::string plugin_name = "PLUGIN";

static const std::string widrowhoff_type = "WIDROWHOFF";
static const std::string copy_type = "COPY";
static const std::string spatial_filter_type = "SPATIALFILTER";
static const std::string custom_type = "CUSTOM";
static const std::string mean_square_error_type = "MEANSQUAREERROR";
static const std::string frechet_distance_type = "FRECHETDISTANCE";
static const std::string feedforward_name = "FEEDFORWARD";
static const std::string feedback_name = "FEEDBACK";
static const std::string recurrent_name = "RECURRENT";
static const std::string readout_name = "READOUT";
static const std::string weights_name = "WEIGHTS";
static const std::string position_name = "POSITION";
static const std::string gaussian_type = "GAUSSIAN";
static const std::string uniform_type = "UNIFORM";
static const std::string tiled_type = "TILED";
static const std::string snippets_type = "SNIPPETS";
static const std::string shuffle_type = "SHUFFLE";
static const std::string punch_type = "PUNCH";
static const std::string reverse_type = "REVERSE";
static const std::string states_type = "STATES";
static const std::string weights_type = "WEIGHTS";
static const std::string performances_type = "PERFORMANCES";
static const std::string scheduling_type = "SCHEDULING";


struct Sequence
{
	std::size_t rows;
	std::size_t cols;
	std::vector<float> elements;
};

static std::map<std::pair<std::string, std::string>, Sequence> sequences_map;



boost::shared_ptr<TRN4CPP::Plugin::Simplified::Interface> simplified;

void TRN4CPP::Plugin::Simplified::initialize(const std::string &library_path, const std::string &name, const std::map<std::string, std::string>  &arguments)
{
	if (simplified)
		throw std::runtime_error("A plugin is already loaded");
	boost::filesystem::path path = library_path;

	path /=  name;

	simplified = boost::dll::import<TRN4CPP::Plugin::Simplified::Interface>(path, "plugin_simplified", boost::dll::load_mode::append_decorations);
	simplified->initialize(arguments);
	simplified->install_variable(std::bind(&TRN4CPP::Simulation::declare, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
}


void TRN4CPP::Simulation::declare(const std::string &label, const std::vector<float> &elements, const std::size_t rows, const std::size_t &cols, const std::string &tag)
{
	auto key = std::make_pair(label, tag);
	if (sequences_map.find(key) != sequences_map.end())
		throw std::invalid_argument("Sequence have already been declared");
	sequences_map[key].rows = rows;
	sequences_map[key].cols = cols;
	sequences_map[key].elements = elements;
}

void TRN4CPP::Simulation::compute(const std::string &filename)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");

	std::cout << "Reading file " << filename << std::endl;
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


	const std::string id_attribute = prefix + "ID";
	const std::string tests_attribute = prefix + "TESTS";
	const std::string trials_attribute = prefix + "TRIALS";
	const std::string type_attribute = prefix + "TYPE";
	const std::string batch_size_attribute = prefix + "BATCH_SIZE";
	const std::string batch_number_attribute = prefix + "BATCH_NUMBER";
	const std::string stimulus_size_attribute = prefix + "STIMULUS_SIZE";
	const std::string reservoir_size_attribute = prefix + "RESERVOIR_SIZE";
	const std::string prediction_size_attribute = prefix + "PREDICTION_SIZE";
	const std::string leak_rate_attribute = prefix + "LEAK_RATE";
	const std::string learning_rate_attribute = prefix + "LEARNING_RATE";
	const std::string initial_state_scale_attribute = prefix + "INITIAL_STATE_SCALE";
	const std::string rows_attribute = prefix + "ROWS";
	const std::string cols_attribute = prefix + "COLS";
	const std::string seed_attribute = prefix + "SEED";
	const std::string rate_attribute = prefix + "RATE";
	const std::string size_attribute = prefix + "SIZE";
	const std::string number_attribute = prefix + "NUMBER";
	const std::string mu_attribute = prefix + "MU";
	const std::string sigma_attribute = prefix + "SIGMA";
	const std::string scale_attribute = prefix + "SCALE";
	const std::string label_attribute = prefix + "LABEL";
	const std::string radius_attribute = prefix + "RADIUS";
	const std::string tag_attribute = prefix + "TAG";
	const std::string a_attribute = prefix + "A";
	const std::string b_attribute = prefix + "B";
	const std::string sparsity_attribute = prefix + "SPARSITY";
	const std::string response_attribute = prefix + "RESPONSE";
	const std::string x_min_attribute = prefix + "x_min";
	const std::string x_max_attribute = prefix + "x_max";
	const std::string y_min_attribute = prefix + "y_min";
	const std::string y_max_attribute = prefix + "y_max";
	const std::string epochs_attribute = prefix + "epochs";
	const std::string snippets_size_attribute = prefix + "snippets_size";
	const std::string time_budget_attribute = prefix + "time_budget";
	const std::string target_attribute = prefix + "target";
	const std::string train_attribute = prefix + "train";
	const std::string generate_attribute = prefix + "generate";
	const std::string prime_attribute = prefix + "prime";
	const std::string initialize_attribute = prefix + "initialize";
	const std::string incoming_attribute = prefix + "incoming";
	const std::string expected_attribute = prefix + "expected";
	const std::string preamble_attribute = prefix + "preamble";
	const std::string repeat_attribute = prefix + "repeat";
	const std::string supplementary_attribute = prefix + "supplementary";
	const std::string filename_attribute = prefix + "filename";
	const std::string path_attribute = prefix + "path";
	const std::string name_attribute = prefix + "name";
	const std::string interface_attribute = prefix + "interface";
	const std::string arguments_attribute = prefix + "arguments";
	std::size_t total_simulations = 0;


	unsigned short experiment_number;
	unsigned short condition_number;
	unsigned int simulation_number;

	try
	{
		bool experiment_done = false;
		experiment_number = 1;
		for (auto property_element : properties)
		{
			if (boost::iequals(property_element.first, experiment_name))
			{
				if (experiment_done)
					throw std::runtime_error("Only one experiment per file is allowed");
				condition_number = 1;
				auto _experiment = property_element.second;
				for (auto experiment_element : _experiment)
				{
					if (boost::iequals(experiment_element.first, plugin_name))
					{
						auto _plugin = experiment_element.second;
						auto interface = _plugin.get_child(interface_attribute).get_value<std::string>();
						auto path = _plugin.get_child(path_attribute).get_value<std::string>();
						auto name = _plugin.get_child(name_attribute).get_value<std::string>();
						std::vector<std::string> arguments_list;

						boost::split(arguments_list, _plugin.get<std::string>(arguments_attribute, ""), boost::is_any_of(",; \t"));
						std::map<std::string, std::string> arguments;
						for (auto argument : arguments_list)
						{
							std::vector<std::string> key_value;
							boost::split(key_value, argument, boost::is_any_of("="));
							if (key_value.size() == 2)
							{
								auto key = key_value[0];
								boost::to_upper(key);
								auto value = key_value[1];
								arguments[key] = value;
							}
						}


						if (boost::iequals(interface, "Simplified"))
						{
							TRN4CPP::Plugin::Simplified::initialize(path, name, arguments);
						}
						else if (boost::iequals(interface, "Custom"))
						{
							TRN4CPP::Plugin::Custom::initialize(path, name, arguments);
						}
						else if (boost::iequals(interface, "Callbacks"))
						{
							TRN4CPP::Plugin::Callbacks::initialize(path, name, arguments);
						}
						else
							throw std::runtime_error("Unexpected plugin interface " + interface);
					}
					else if (boost::iequals(experiment_element.first, data_name))
					{
						auto _data = experiment_element.second;

						for (auto data_element : _data)
						{
							if (boost::iequals(data_element.first, variable_name))
							{
								auto _variable = data_element.second;
								auto label = _variable.get_child(label_attribute).get_value<std::string>();
								auto tag = _variable.get(tag_attribute, "");
								simplified->callback_variable(label, tag);
							}

						}
						simplified.reset();
					}
					else if (boost::iequals(experiment_element.first, condition_name))
					{
						auto _simulation = experiment_element.second;
						const std::size_t batch_number = _simulation.get <std::size_t>(batch_number_attribute, 1);
						const std::size_t batch_size = _simulation.get <std::size_t>(batch_size_attribute, 1);
						for (simulation_number = 1; simulation_number <= batch_number; simulation_number++)
						{
							bool reservoir_initialized = false;
							bool loop_initialized = false;
							bool scheduler_initialized = false;
							bool readout_mean_square_error_initialized = false;
							bool position_mean_square_error_initialized = false;
							bool readout_frechet_distance_initialized = false;
							bool position_frechet_distance_initialized = false;
							bool readout_custom_initialized = false;
							bool position_custom_initialized = false;
							bool states_initialized = false;
							bool weights_initialized = false;
							bool performances_initialized = false;
							bool scheduling_initialized = false;

		
							std::size_t stimulus_size;
							std::size_t reservoir_size;
							std::size_t prediction_size;
							unsigned long long id;
							TRN4CPP::Simulation::encode(experiment_number, condition_number, simulation_number, id);

							frontend->allocate(id);
							//
							bool configured = false;
							bool declared = false;
							for (auto simulation_element : _simulation)
							{
								/// DECLARATION
								if (boost::iequals(simulation_element.first, declaration_name))
								{
									if (!configured)
										throw std::runtime_error("Simulation  #" + std::to_string(id) + " must be configured first");
									auto _declaration = simulation_element.second;
									for (auto declaration_element : _declaration)
									{
										/// SEQUENCES
										if (boost::iequals(declaration_element.first, sequence_name))
										{
											auto _sequence = declaration_element.second;
											auto label = _sequence.get_child(label_attribute).get_value<std::string>();
											for (auto sequence_element : _sequence)
											{
												auto sequence_element_name = sequence_element.first;
												if (boost::iequals(sequence_element.first, tag_name))
												{
													auto _tag = sequence_element.second;
													auto tag = _tag.get_value<std::string>();
													auto key = std::make_pair(label, tag);
													if (sequences_map.find(key) == sequences_map.end())
														throw std::runtime_error("Sequence having label " + label + " and tag " + tag + "does not exist");
													auto data = sequences_map[key];
													//
													frontend->declare_sequence(id, label, tag, data.elements, data.rows);
												}
											}
										}
										/// SETS
										else if (boost::iequals(declaration_element.first, set_name))
										{
											auto set = declaration_element.second;
											auto label = set.get_child(label_attribute).get_value<std::string>();
											std::vector<std::string> labels;
											std::vector<std::string> tags;
											for (auto set_element : set)
											{
												auto set_element_name = set_element.first;
												boost::to_upper(set_element_name);
												if (set_element_name == label_name)
												{
													auto label = set_element.second;
													labels.push_back(label.get_value<std::string>());
												}
												else if (set_element_name == tag_name)
												{
													auto tag = set_element.second;
													tags.push_back(tag.get_value<std::string>());
												}
											}
											for (auto tag : tags)
											{
												//
												frontend->declare_set(id, label, tag, labels);

											}

										}
									}
									declared = true;
								}
								/// CONFIGURATION
								else if (boost::iequals(simulation_element.first, configuration_name))
								{
									if (configured)
										throw std::runtime_error("Simulation #" + std::to_string(id) + " is already configured");
									
									frontend->configure_begin(id);
									//
									auto _configuration = simulation_element.second;
									for (auto configuration_element : _configuration)
									{
										if (boost::iequals(configuration_element.first, model_name))
										{
											auto _model = configuration_element.second;
											for (auto model_element : _model)
											{
												/// RESERVOIR
												if (boost::iequals(model_element.first, reservoir_name))
												{
													if (reservoir_initialized)
														throw std::runtime_error("Only one reservoir per simulation is allowed");

													auto _reservoir = model_element.second;

											
													stimulus_size = _reservoir.get_child(stimulus_size_attribute).get_value<std::size_t>();
													reservoir_size = _reservoir.get_child(reservoir_size_attribute).get_value<std::size_t>();
													prediction_size = _reservoir.get_child(prediction_size_attribute).get_value<std::size_t>();
													auto seed = simulation_number + _reservoir.get_child(seed_attribute).get_value<unsigned long>();
													auto leak_rate = _reservoir.get_child(leak_rate_attribute).get_value<float>();
													auto initial_state_scale = _reservoir.get_child(initial_state_scale_attribute).get_value<float>();
													auto reservoir_type = _reservoir.get_child(type_attribute).get_value<std::string>();

													if (boost::iequals(reservoir_type, widrowhoff_type))
													{
														auto learning_rate = _reservoir.get_child(learning_rate_attribute).get_value<float>();
														frontend->configure_reservoir_widrow_hoff(id, stimulus_size, prediction_size, reservoir_size, leak_rate, initial_state_scale, learning_rate, seed, batch_size);
														//
													}
													else
														throw std::runtime_error("Unexpected reservoir type " + reservoir_type);
													bool feedback_initialized = false;
													bool feedforward_initialized = false;
													bool recurrent_initialized = false;
													bool readout_initialized = false;
													for (auto reservoir_element : _reservoir)
													{
														if (boost::iequals(reservoir_element.first, weights_name))
														{
															auto _weights = reservoir_element.second;
															auto target = _weights.get_child(target_attribute).get_value<std::string>();
															auto type = _weights.get_child(type_attribute).get_value<std::string>();
															if (boost::iequals(target, feedforward_name))
															{
																if (boost::iequals(type, gaussian_type))
																{
																	auto mu = _weights.get_child(mu_attribute).get_value<float>();
																	auto sigma = _weights.get_child(sigma_attribute).get_value<float>();
																	frontend->configure_feedforward_gaussian(id, mu, sigma);
																	
																	feedforward_initialized = true;
																}
																else if (boost::iequals(type, uniform_type))
																{
																	auto a = _weights.get_child(a_attribute).get_value<float>();
																	auto b = _weights.get_child(b_attribute).get_value<float>();
																	auto sparsity = _weights.get_child(sparsity_attribute).get_value<float>();
																	frontend->configure_feedforward_uniform(id, a, b, sparsity);
																	
																	feedforward_initialized = true;
																}
																else if (boost::iequals(type, custom_type))
																{
																	if (!on_feedforward)
																		throw std::runtime_error("Feedforward functor must be installed first");
																	frontend->install_feedforward(id, on_feedforward);
																	frontend->configure_feedforward_custom(id);
																	//
																	feedforward_initialized = true;
																}
															}
															else if (boost::iequals(target, feedback_name))
															{
																if (boost::iequals(type, gaussian_type))
																{
																	auto mu = _weights.get_child(mu_attribute).get_value<float>();
																	auto sigma = _weights.get_child(sigma_attribute).get_value<float>();
																	frontend->configure_feedback_gaussian(id, mu, sigma);
																	//
																	feedback_initialized = true;
																}
																else if (boost::iequals(type, uniform_type))
																{
																	auto a = _weights.get_child(a_attribute).get_value<float>();
																	auto b = _weights.get_child(b_attribute).get_value<float>();
																	auto sparsity = _weights.get_child(sparsity_attribute).get_value<float>();
																	frontend->configure_feedback_uniform(id, a, b, sparsity);
																	//
																	feedback_initialized = true;
																}
																else if (boost::iequals(type, custom_type))
																{
																	if (!on_feedback)
																		throw std::runtime_error("Feedback functor must be installed first");
																	frontend->install_feedback(id, on_feedback);
																	frontend->configure_feedback_custom(id);
																	//
																	feedback_initialized = true;
																}
															}
															else if (boost::iequals(target, recurrent_name))
															{
																if (boost::iequals(type, gaussian_type))
																{
																	auto mu = _weights.get_child(mu_attribute).get_value<float>();
																	auto sigma = _weights.get_child(sigma_attribute).get_value<float>();
																	frontend->configure_recurrent_gaussian(id, mu, sigma);
																	//
																	recurrent_initialized = true;
																}
																else if (boost::iequals(type, uniform_type))
																{
																	auto a = _weights.get_child(a_attribute).get_value<float>();
																	auto b = _weights.get_child(b_attribute).get_value<float>();
																	auto sparsity = _weights.get_child(sparsity_attribute).get_value<float>();
																	frontend->configure_recurrent_uniform(id, a, b, sparsity);
																	//
																	recurrent_initialized = true;
																}
																else if (boost::iequals(type, custom_type))
																{
																	if (!on_recurrent)
																		throw std::runtime_error("Recurrent functor must be installed first");
																	frontend->install_recurrent(id, on_recurrent);
																	frontend->configure_recurrent_custom(id);
																	//
																	recurrent_initialized = true;
																}
															}
															else if (boost::iequals(target, readout_name))
															{
																auto _initializer = reservoir_element.second;
																auto type = _initializer.get_child(type_attribute).get_value<std::string>();
																if (boost::iequals(type, gaussian_type))
																{
																	auto mu = _initializer.get_child(mu_attribute).get_value<float>();
																	auto sigma = _initializer.get_child(sigma_attribute).get_value<float>();
																	frontend->configure_readout_gaussian(id, mu, sigma);
																	//
																	readout_initialized = true;
																}
																else if (boost::iequals(type, uniform_type))
																{
																	auto a = _initializer.get_child(a_attribute).get_value<float>();
																	auto b = _initializer.get_child(b_attribute).get_value<float>();
																	auto sparsity = _initializer.get_child(sparsity_attribute).get_value<float>();
																	frontend->configure_readout_uniform(id, a, b, sparsity);
																	//
																	readout_initialized = true;
																}
																else if (boost::iequals(type, custom_type))
																{
																	if (!on_readout)
																		throw std::runtime_error("Readout functor must be installed first");
																	frontend->install_readout(id, on_readout);
																	frontend->configure_readout_custom(id);
																	//
																	readout_initialized = true;
																}
															}
														}
													}

													if (!feedforward_initialized)
														throw std::runtime_error("Feedforward was not initialized");
													if (!feedback_initialized)
														throw std::runtime_error("Feedback was not initialized");
													if (!recurrent_initialized)
														throw std::runtime_error("Recurrent was not initialized");
													if (!readout_initialized)
														throw std::runtime_error("Readout was not initialized");
													reservoir_initialized = true;
												}
												/// LOOP				
												else if (boost::iequals(model_element.first, loop_name))
												{
													if (loop_initialized)
														throw std::runtime_error("Only one loop per simulation is allowed");
													if (!reservoir_initialized)
														throw std::runtime_error("Reservoir must be initialized first");

													auto _loop = model_element.second;
													auto loop_type = _loop.get_child(type_attribute).get_value<std::string>();
													if (boost::iequals(loop_type, copy_type))
													{
														frontend->configure_loop_copy(id, batch_size, stimulus_size);
														//
													}
													else if (boost::iequals(loop_type, custom_type))
													{
														if (!on_stimulus)
															throw std::runtime_error("Stimulus callback must be installed first");
														frontend->install_stimulus(id, on_stimulus);
														frontend->configure_loop_custom(id, batch_size, stimulus_size);
														//
													}
													else if (boost::iequals(loop_type, spatial_filter_type))
													{
														if (!on_stimulus)
															throw std::runtime_error("Stimulus callback must be installed first");
														if (!on_position)
															throw std::runtime_error("Position callback must be installed first");

														auto seed = simulation_number + _loop.get_child(seed_attribute).get_value<unsigned long>();
														auto sigma = _loop.get_child(sigma_attribute).get_value<float>();
														auto scale = _loop.get_child(scale_attribute).get_value<float>();
														auto radius = _loop.get_child(radius_attribute).get_value<float>();
														auto tag = _loop.get_child(tag_attribute).get_value<std::string>();
														auto response = _loop.get_child(response_attribute).get_value<std::string>();
														auto x_min = _loop.get_child(x_min_attribute).get_value<float>();
														auto x_max = _loop.get_child(x_max_attribute).get_value<float>();
														auto y_min = _loop.get_child(y_min_attribute).get_value<float>();
														auto y_max = _loop.get_child(y_max_attribute).get_value<float>();
														auto x = std::make_pair(x_min, x_max);
														auto y = std::make_pair(y_min, y_max);
														auto key = std::make_pair(response, "");
														if (sequences_map.find(key) == sequences_map.end())
															throw std::runtime_error("Can't find response matrix " + response);
														auto &response_matrix = sequences_map[key];
														auto cols = response_matrix.cols;
														auto rows = response_matrix.rows / stimulus_size;
														auto firing_rate_map = response_matrix.elements;
														frontend->install_stimulus(id, on_stimulus);
														frontend->install_position(id, on_position);
														frontend->configure_loop_spatial_filter(id, batch_size, stimulus_size, seed, rows, cols, x, y, firing_rate_map, sigma, radius, scale, tag);
														//
													}
												}
												/// SCHEDULER
												else if (boost::iequals(model_element.first, scheduler_name))
												{
													auto _scheduler = model_element.second;
													auto scheduler_type = _scheduler.get_child(type_attribute).get_value<std::string>();
													if (boost::iequals(scheduler_type, tiled_type))
													{
														auto epochs = _scheduler.get_child(epochs_attribute).get_value<unsigned int>();
														frontend->configure_scheduler_tiled(id, epochs);
														//
													}
													else if (boost::iequals(scheduler_type, snippets_type))
													{
														auto seed = simulation_number + _scheduler.get_child(seed_attribute).get_value<unsigned long>();
														auto snippets_size = _scheduler.get_child(snippets_size_attribute).get_value<std::size_t>();
														auto time_budget = _scheduler.get_child(time_budget_attribute).get_value<std::size_t>();
														auto tag = _scheduler.get(tag_attribute, DEFAULT_TAG);
														frontend->configure_scheduler_snippets(id, seed, snippets_size, time_budget, tag);
														//
													}
													else if (boost::iequals(scheduler_type, custom_type))
													{
														auto seed = simulation_number + _scheduler.get_child(seed_attribute).get_value<unsigned long>();
														auto tag = _scheduler.get_child(seed_attribute).get_value<std::string>(DEFAULT_TAG);
														if (!on_scheduler)
															throw std::runtime_error("Scheduler callback must be installed first");
														frontend->install_scheduler(id, on_scheduler);
														frontend->configure_scheduler_custom(id, seed, tag);
														//
													}
													bool mutator_initialized = false;
													for (auto scheduler_element : _scheduler)
													{
														/// MUTATOR
														if (boost::iequals(scheduler_element.first, mutator_name))
														{

															auto _mutator = scheduler_element.second;
															auto mutator_type = _mutator.get_child(type_attribute).get_value<std::string>();
															if (boost::iequals(mutator_type, reverse_type))
															{
																auto seed = simulation_number + _mutator.get_child(seed_attribute).get_value<unsigned long>();
																auto rate = _mutator.get_child(rate_attribute).get_value<float>();
																auto size = _mutator.get_child(size_attribute).get_value<std::size_t>();
																//auto number = _mutator.get_child(number_attribute).get_value<std::size_t>();
																frontend->configure_mutator_reverse(id, seed, rate, size);
																//
															}
															else if (boost::iequals(mutator_type, punch_type))
															{
																auto seed = simulation_number + _mutator.get_child(seed_attribute).get_value<unsigned long>();
																auto rate = _mutator.get_child(rate_attribute).get_value<float>();
																auto size = _mutator.get_child(size_attribute).get_value<std::size_t>();
																auto number = _mutator.get_child(number_attribute).get_value<std::size_t>();
																frontend->configure_mutator_punch(id, seed, rate, size, number);
																//
															}
															else if (boost::iequals(mutator_type, shuffle_type))
															{
																auto seed = simulation_number + _mutator.get_child(seed_attribute).get_value<unsigned long>();

																frontend->configure_mutator_shuffle(id, seed);
																//
															}
															else if (boost::iequals(mutator_type, custom_type))
															{
																if (mutator_initialized)
																	throw std::runtime_error("Custom mutator is already initialized");
																if (!on_mutator)
																	throw std::runtime_error("Mutator callback must be installed first");
																auto seed = simulation_number + _mutator.get_child(seed_attribute).get_value<unsigned long>();

																frontend->install_mutator(id, on_mutator);
																frontend->configure_mutator_custom(id, seed);
																//
																mutator_initialized = true;
															}
														}
													}
												}
											}
										}
										else if (boost::iequals(configuration_element.first, result_name))
										{
											auto _result = configuration_element.second;
											for (auto result_element : _result)
											{
												/// MEASUREMENT
												if (boost::iequals(result_element.first, measurement_name))
												{
													auto _measurement = result_element.second;
													auto measurement_target = _measurement.get_child(target_attribute).get_value<std::string>();
													if (boost::iequals(measurement_target, readout_name))
													{
														auto measurement_type = _measurement.get_child(type_attribute).get_value<std::string>();
														if (boost::iequals(measurement_type, mean_square_error_type))
														{
															if (readout_mean_square_error_initialized)
																throw std::runtime_error("Readout mean square error is already initialized");
															if (!on_measurement_readout_mean_square_error)
																throw std::runtime_error("Readout mean square error callback must be installed first");
															frontend->install_measurement_readout_mean_square_error(id, on_measurement_readout_mean_square_error);
															frontend->configure_measurement_readout_mean_square_error(id, batch_size);
															//
															readout_mean_square_error_initialized = true;
														}
														else if (boost::iequals(measurement_type, frechet_distance_type))
														{
															if (readout_frechet_distance_initialized)
																throw std::runtime_error("Readout frechet distance is already initialized");
															if (!on_measurement_readout_frechet_distance)
																throw std::runtime_error("Readout frechet distance callback must be installed first");
															frontend->install_measurement_readout_frechet_distance(id, on_measurement_readout_frechet_distance);
															frontend->configure_measurement_readout_frechet_distance(id, batch_size);
															//
															readout_frechet_distance_initialized = true;
														}
														else if (boost::iequals(measurement_type, custom_type))
														{
															if (readout_custom_initialized)
																throw std::runtime_error("Readout custom is already initialized");
															if (!on_measurement_readout_raw)
																throw std::runtime_error("Readout custom callback must be installed first");
															frontend->install_measurement_readout_custom(id, on_measurement_readout_raw);
															frontend->configure_measurement_readout_custom(id, batch_size);
															//
															readout_custom_initialized = true;
														}
													}
													else if (boost::iequals(measurement_target, position_name))
													{
														auto measurement_type = _measurement.get_child(type_attribute).get_value<std::string>();
														if (boost::iequals(measurement_type, mean_square_error_type))
														{
															if (position_mean_square_error_initialized)
																throw std::runtime_error("Position mean square error is already initialized");
															if (!on_measurement_position_mean_square_error)
																throw std::runtime_error("Position mean square error callback must be installed first");
															frontend->install_measurement_position_mean_square_error(id, on_measurement_position_mean_square_error);
															frontend->configure_measurement_position_mean_square_error(id, batch_size);
															//
															position_mean_square_error_initialized = true;
														}
														else if (boost::iequals(measurement_type, frechet_distance_type))
														{
															if (position_frechet_distance_initialized)
																throw std::runtime_error("Position frechet distance is already initialized");
															if (!on_measurement_position_frechet_distance)
																throw std::runtime_error("Position frechet distance callback must be installed first");
															frontend->install_measurement_position_frechet_distance(id, on_measurement_position_frechet_distance);
															frontend->configure_measurement_position_frechet_distance(id, batch_size);
															//
															position_frechet_distance_initialized = true;
														}
														else if (boost::iequals(measurement_type, custom_type))
														{
															if (position_custom_initialized)
																throw std::runtime_error("Position custom is already initialized");
															if (!on_measurement_position_raw)
																throw std::runtime_error("Position custom callback must be installed first");
															frontend->install_measurement_position_custom(id, on_measurement_position_raw);
															frontend->configure_measurement_position_custom(id, batch_size);
															//
															position_custom_initialized = true;
														}
													}
												}
												/// RECORDING
												else if (boost::iequals(result_element.first, recording_name))
												{
													auto _measurement = result_element.second;
													auto measurement_target = _measurement.get_child(target_attribute).get_value<std::string>();
													if (boost::iequals(measurement_target, states_type))
													{
														auto train = _measurement.get_child(train_attribute).get_value<bool>();
														auto prime = _measurement.get_child(prime_attribute).get_value<bool>();
														auto generate = _measurement.get_child(generate_attribute).get_value<bool>();

														if (!on_states)
															throw std::runtime_error("States callback must be installed first");
														if (!train && !prime && !generate)
														{
															std::cerr << "States decorator won't be installed because no experiment stage (train, prime, generate) is selected" << std::endl;
														}
														else
														{
															frontend->install_states(id, on_states);
															frontend->setup_states(id, train, prime, generate);
															//
														}
													}
													else if (boost::iequals(measurement_target, weights_type))
													{
														auto train = _measurement.get_child(train_attribute).get_value<bool>();
														auto initialization = _measurement.get_child(initialize_attribute).get_value<bool>();

														if (!on_weights)
															throw std::runtime_error("Weights callback must be installed first");
														if (!train && !initialization)
														{
															std::cerr << "Weights decorator won't be installed because no experiment stage (initializen, train) is selected" << std::endl;
														}
														else
														{
															frontend->install_weights(id, on_weights);
															frontend->setup_weights(id, initialization, train);
															//
														}
													}
													else if (boost::iequals(measurement_target, performances_type))
													{
														auto train = _measurement.get_child(train_attribute).get_value<bool>();
														auto prime = _measurement.get_child(prime_attribute).get_value<bool>();
														auto generate = _measurement.get_child(generate_attribute).get_value<bool>();

														if (!on_performances)
															throw std::runtime_error("Performances callback must be installed first");
														if (!train && !prime && !generate)
														{
															std::cerr << "Performances decorator won't be installed because no experiment stage (train, prime, generate) is selected" << std::endl;
														}
														else
														{
															frontend->install_performances(id, on_performances);
															frontend->setup_performances(id, train, prime, generate);
															//
														}
													}
													else if (boost::iequals(measurement_target, scheduling_type))
													{
														if (!on_scheduling)
															throw std::runtime_error("Scheduling callback must be installed first");
														frontend->install_scheduling(id, on_scheduling);
														frontend->setup_scheduling(id);
														//
													}
												}
											}

										}
									}
									
									frontend->configure_end(id);

									//
									configured = true;
								}
								/// EXECUTION
								else if (boost::iequals(simulation_element.first, execution_name))
								{
									if (!configured)
										throw std::runtime_error("Simulation #" + std::to_string(id) + " is not configured");
									if (!declared)
										throw std::runtime_error("Simulation #" + std::to_string(id) + " have no data declared");
									auto _execution = simulation_element.second;

									for (auto execution_element : _execution)
									{
										std::size_t trial_number = 1;
										if (boost::iequals(execution_element.first, trial_name))
										{
											auto _trial = execution_element.second;
											bool trained = false;
											for (auto trial_element : _trial)
											{
												if (boost::iequals(trial_element.first, train_name))
												{
													if (trained)
														throw std::runtime_error("Simulation #" + std::to_string(id) + " is already trained for trial #" + std::to_string(trial_number));
													auto _train = trial_element.second;
													auto label = _train.get_child(label_attribute).get_value<std::string>();
													auto incoming = _train.get_child(incoming_attribute).get_value<std::string>();
													auto expected = _train.get_child(expected_attribute).get_value<std::string>();
													
													frontend->train(id, label, incoming, expected);

													trained = true;
												}
												else if (boost::iequals(trial_element.first, test_name))
												{
													if (!trained)
														throw std::runtime_error("Simulation #" + std::to_string(id) + " must be trained before test");
													auto _test = trial_element.second;
													auto preamble = _test.get_child(preamble_attribute).get_value<unsigned int>();
													auto supplementary = _test.get_child(supplementary_attribute).get_value<unsigned int>();
													auto label = _test.get_child(label_attribute).get_value<std::string>();
													auto incoming = _test.get_child(incoming_attribute).get_value<std::string>();
													auto expected = _test.get_child(expected_attribute).get_value<std::string>();
													auto repeat = _test.get_child(repeat_attribute).get_value<std::size_t>();
													for (std::size_t k = 0; k < repeat; k++)
													{
														
														frontend->test(id, label, incoming, expected, preamble, supplementary);
													}

												}
											}
											trial_number++;
										}
									}

								}
							}
							
							frontend->deallocate(id);
						}
						condition_number++;
					}
				}
				experiment_done = true;
				experiment_number++;
			}
		}
		frontend->join();
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}