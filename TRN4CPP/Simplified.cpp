#include "stdafx.h"
#include "Simplified.h"
#include "Engine/Frontend.h"
#include "Callbacks.h"
#include "Custom.h"
#include "Extended.h"
#include "Sequences.h"
#include "Search.h"

#include "Helper/Logger.h"


static const float DEFAULT_SCALE = 1.0f;
static const float DEFAULT_MU = 0.0f;
static const float DEFAULT_SIGMA = 1.0f;
static const float DEFAULT_A = -1.0f;
static const float DEFAULT_B = 1.0f;
static const float DEFAULT_SPARSITY = 0.0f;
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

template<typename Type>
static Type default_value()
{
	return static_cast<Type>(0);
}
template<>
static std::string default_value<std::string>()
{
	return "";
}

template<typename Type> 
static Type get_variable(const boost::property_tree::iptree &node, const std::string &label, const unsigned short &condition_number, const unsigned int &simulation_number, const Type &default = default_value<Type>())
{
	auto child = node.get_child_optional(label);
	if (!child)
		return default;
	auto value = child->get_value<std::string>();

	if (value.at(0) == '@')
	{
		value = TRN4CPP::Search::retrieve(condition_number, simulation_number, value.substr(1));
	}
	
	return boost::lexical_cast<Type>(value);
}

template<typename Type>
static Type get_attribute(const boost::property_tree::iptree &node, const std::string &label, const Type &default = default_value<Type>())
{
	auto child = node.get_child_optional(label);
	if (!child)
		return default;
	return child->get_value<Type>(default);
}

void TRN4CPP::Simulation::compute(const std::string &filename)
{
	INFORMATION_LOGGER <<   "Reading file " << filename ;
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


	const std::string id_attribute = prefix + "id";
	const std::string tests_attribute = prefix + "tests";
	const std::string trials_attribute = prefix + "trials";
	const std::string type_attribute = prefix + "type";
	const std::string batch_size_attribute = prefix + "batch_size";
	const std::string batch_number_attribute = prefix + "batch_number";
	const std::string stimulus_size_attribute = prefix + "stimulus_size";
	const std::string reservoir_size_attribute = prefix + "reservoir_size";
	const std::string prediction_size_attribute = prefix + "prediction_size";
	const std::string leak_rate_attribute = prefix + "leak_rate";
	const std::string learning_rate_attribute = prefix + "learning_rate";
	const std::string initial_state_scale_attribute = prefix + "initial_state_scale";
	const std::string rows_attribute = prefix + "rows";
	const std::string cols_attribute = prefix + "cols";
	const std::string seed_attribute = prefix + "seed";
	const std::string rate_attribute = prefix + "rate";
	const std::string size_attribute = prefix + "size";
	const std::string number_attribute = prefix + "number";
	const std::string mu_attribute = prefix + "mu";
	const std::string sigma_attribute = prefix + "sigma";
	const std::string scale_attribute = prefix + "scale";
	const std::string label_attribute = prefix + "label";
	const std::string radius_attribute = prefix + "radius";
	const std::string tag_attribute = prefix + "tag";
	const std::string a_attribute = prefix + "a";
	const std::string b_attribute = prefix + "b";
	const std::string sparsity_attribute = prefix + "sparsity";
	const std::string response_attribute = prefix + "response";
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
	const std::string reset_attribute = prefix + "reset";
	const std::string autonomous_attribute = prefix + "autonomous";
	const std::string supplementary_attribute = prefix + "supplementary";
	const std::string filename_attribute = prefix + "filename";
	const std::string path_attribute = prefix + "path";
	const std::string name_attribute = prefix + "name";
	const std::string interface_attribute = prefix + "interface";
	const std::string arguments_attribute = prefix + "arguments";
	std::size_t total_simulations = 0;


	unsigned short condition_number;
	unsigned int simulation_number;

	try
	{
		bool experiment_done = false;
		for (auto property_element : properties)
		{
			if (boost::iequals(property_element.first, experiment_name))
			{
				if (experiment_done)
					throw std::runtime_error("Only one experiment per file is allowed");
				condition_number = 1;
				simulation_number = 1;
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


						if (boost::iequals(interface, "Sequences"))
						{
							TRN4CPP::Plugin::Sequences::initialize(path, name, arguments);
						}
						else if (boost::iequals(interface, "Search"))
						{
							TRN4CPP::Plugin::Search::initialize(path, name, arguments);
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
								if (label.empty())
									throw std::runtime_error("Unexpected empty label");
								auto tag = _variable.get(tag_attribute, TRN4CPP::Sequences::DEFAULT_TAG);

								TRN4CPP::Sequences::fetch(label, tag);
							}

						}
					}
					else if (boost::iequals(experiment_element.first, condition_name))
					{
						auto _simulation = experiment_element.second;

						std::size_t batch_size = _simulation.get<std::size_t>(batch_size_attribute, 1);
						std::size_t batch_number = _simulation.get<std::size_t>(batch_number_attribute, 1);

						TRN4CPP::Search::begin(condition_number, simulation_number, batch_number, batch_size);
						do
						{
							for (unsigned int k = 0; k < TRN4CPP::Search::size(condition_number); k++)
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
								TRN4CPP::Simulation::encode(0, condition_number, simulation_number, id);
								TRN4CPP::Simulation::allocate(id);

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
														std::size_t rows, cols;
														std::vector<float> elements;
														TRN4CPP::Sequences::retrieve(label, tag, elements, rows, cols);
												
														//
														TRN4CPP::Simulation::declare_sequence(id, label, tag, elements, rows);
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
													TRN4CPP::Simulation::declare_set(id, label, tag, labels);
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
										TRN4CPP::Simulation::configure_begin(id);
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


														stimulus_size = get_variable<std::size_t>(_reservoir, stimulus_size_attribute, condition_number, simulation_number);
														reservoir_size = get_variable<std::size_t>(_reservoir, reservoir_size_attribute, condition_number, simulation_number);
														prediction_size = get_variable<std::size_t>(_reservoir, prediction_size_attribute, condition_number, simulation_number);
														auto seed = simulation_number + get_variable<unsigned long>(_reservoir, seed_attribute, condition_number, simulation_number);
														auto leak_rate = get_variable<float>(_reservoir, leak_rate_attribute, condition_number, simulation_number);
														auto initial_state_scale = get_variable<float>(_reservoir, initial_state_scale_attribute, condition_number, simulation_number);
														auto reservoir_type = _reservoir.get_child(type_attribute).get_value<std::string>();

														if (boost::iequals(reservoir_type, widrowhoff_type))
														{
															auto learning_rate = get_variable<float>(_reservoir, learning_rate_attribute, condition_number, simulation_number);
																
															TRN4CPP::Simulation::Reservoir::WidrowHoff::configure(id, stimulus_size, prediction_size, reservoir_size, leak_rate, initial_state_scale, learning_rate, seed, batch_size);
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
															
																auto scale = get_variable(_weights, scale_attribute, condition_number, simulation_number, DEFAULT_SCALE);
														
																if (boost::iequals(target, feedforward_name))
																{
																	if (boost::iequals(type, gaussian_type))
																	{
																		auto mu = get_variable(_weights, mu_attribute, condition_number, simulation_number, DEFAULT_MU);
																		auto sigma = get_variable(_weights, sigma_attribute, condition_number, simulation_number, DEFAULT_SIGMA);
																		auto sparsity = get_variable(_weights, sparsity_attribute, condition_number, simulation_number, DEFAULT_SPARSITY);

																		TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Gaussian::configure(id, mu, sigma * scale, sparsity);

																		feedforward_initialized = true;
																	}
																	else if (boost::iequals(type, uniform_type))
																	{
																		auto a = get_variable(_weights, a_attribute, condition_number, simulation_number, DEFAULT_A);
																		auto b = get_variable(_weights, b_attribute, condition_number, simulation_number, DEFAULT_B);
																		auto sparsity = get_variable(_weights, sparsity_attribute, condition_number, simulation_number, DEFAULT_SPARSITY);


																		TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Uniform::configure(id, a * scale, b * scale, sparsity);

																		feedforward_initialized = true;
																	}
																	else if (boost::iequals(type, custom_type))
																	{
																		TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::configure(id);

																		//
																		feedforward_initialized = true;
																	}
																}
																else if (boost::iequals(target, feedback_name))
																{
																	if (boost::iequals(type, gaussian_type))
																	{
																		auto mu = get_variable(_weights, mu_attribute, condition_number, simulation_number, DEFAULT_MU);
																		auto sigma = get_variable(_weights, sigma_attribute, condition_number, simulation_number, DEFAULT_SIGMA);
																		auto sparsity = get_variable(_weights, sparsity_attribute, condition_number, simulation_number, DEFAULT_SPARSITY);


																		TRN4CPP::Simulation::Reservoir::Weights::Feedback::Gaussian::configure(id, mu, sigma * scale, sparsity);
																		//
																		feedback_initialized = true;
																	}
																	else if (boost::iequals(type, uniform_type))
																	{
																		auto a = get_variable(_weights, a_attribute, condition_number, simulation_number, DEFAULT_A);
																		auto b = get_variable(_weights, b_attribute, condition_number, simulation_number, DEFAULT_B);
																		auto sparsity = get_variable(_weights, sparsity_attribute, condition_number, simulation_number, DEFAULT_SPARSITY);

																		TRN4CPP::Simulation::Reservoir::Weights::Feedback::Uniform::configure(id, a * scale, b * scale, sparsity);
																		//
																		feedback_initialized = true;
																	}
																	else if (boost::iequals(type, custom_type))
																	{
																		TRN4CPP::Simulation::Reservoir::Weights::Feedback::Custom::configure(id);
																		//
																		feedback_initialized = true;
																	}
																}
																else if (boost::iequals(target, recurrent_name))
																{
										
																	if (boost::iequals(type, gaussian_type))
																	{
																		auto mu = get_variable(_weights, mu_attribute, condition_number, simulation_number, DEFAULT_MU);
																		auto sigma = get_variable(_weights, sigma_attribute, condition_number, simulation_number, DEFAULT_SIGMA);
																		auto sparsity = get_variable(_weights, sparsity_attribute, condition_number, simulation_number, DEFAULT_SPARSITY);

																		TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Gaussian::configure(id, mu, sigma * scale, sparsity);
																		//
																		recurrent_initialized = true;
																	}
																	else if (boost::iequals(type, uniform_type))
																	{
																		auto a = get_variable(_weights, a_attribute, condition_number, simulation_number, DEFAULT_A);
																		auto b = get_variable(_weights, b_attribute, condition_number, simulation_number, DEFAULT_B);
																		auto sparsity = get_variable(_weights, sparsity_attribute, condition_number, simulation_number, DEFAULT_SPARSITY);

																		TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Uniform::configure(id, a * scale, b * scale, sparsity);
																		//
																		recurrent_initialized = true;
																	}
																	else if (boost::iequals(type, custom_type))
																	{
																		TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::configure(id);
																		//
																		recurrent_initialized = true;
																	}
																}
																else if (boost::iequals(target, readout_name))
																{
																	if (boost::iequals(type, gaussian_type))
																	{
																		auto mu = get_variable(_weights, mu_attribute, condition_number, simulation_number, DEFAULT_MU);
																		auto sigma = get_variable(_weights, sigma_attribute, condition_number, simulation_number, DEFAULT_SIGMA);
																		auto sparsity = get_variable(_weights, sparsity_attribute, condition_number, simulation_number, DEFAULT_SPARSITY);

																		TRN4CPP::Simulation::Reservoir::Weights::Readout::Gaussian::configure(id, mu, sigma * scale, sparsity);
																		//
																		readout_initialized = true;
																	}
																	else if (boost::iequals(type, uniform_type))
																	{
																		auto a = get_variable(_weights, a_attribute, condition_number, simulation_number, DEFAULT_A);
																		auto b = get_variable(_weights, b_attribute, condition_number, simulation_number, DEFAULT_B);
																		auto sparsity = get_variable(_weights, sparsity_attribute, condition_number, simulation_number, DEFAULT_SPARSITY);

																		TRN4CPP::Simulation::Reservoir::Weights::Readout::Uniform::configure(id, a * scale, b * scale, sparsity);
																		//
																		readout_initialized = true;
																	}
																	else if (boost::iequals(type, custom_type))
																	{
																		TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::configure(id);
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
															TRN4CPP::Simulation::Loop::Copy::configure(id, batch_size, stimulus_size);
															//
														}
														else if (boost::iequals(loop_type, custom_type))
														{
															TRN4CPP::Simulation::Loop::Custom::configure(id, batch_size, stimulus_size);
															//
														}
														else if (boost::iequals(loop_type, spatial_filter_type))
														{
															auto seed = simulation_number + get_variable<unsigned long>(_loop, seed_attribute, condition_number, simulation_number);
															
															auto sigma = get_variable<float>(_loop, sigma_attribute, condition_number, simulation_number);
															auto scale = get_variable<float>(_loop, scale_attribute, condition_number, simulation_number);
															auto radius = get_variable<float>(_loop, radius_attribute, condition_number, simulation_number);

															auto tag = _loop.get_child(tag_attribute).get_value<std::string>();
															auto response = _loop.get_child(response_attribute).get_value<std::string>();
															auto x_min = _loop.get_child(x_min_attribute).get_value<float>();
															auto x_max = _loop.get_child(x_max_attribute).get_value<float>();
															auto y_min = _loop.get_child(y_min_attribute).get_value<float>();
															auto y_max = _loop.get_child(y_max_attribute).get_value<float>();
															auto x = std::make_pair(x_min, x_max);
															auto y = std::make_pair(y_min, y_max);
															auto key = std::make_pair(response, "");
															std::vector<float> firing_rate_map;
															std::size_t firing_rate_map_rows, firing_rate_map_cols;
															TRN4CPP::Sequences::retrieve(response, TRN4CPP::Sequences::DEFAULT_TAG, firing_rate_map, firing_rate_map_rows, firing_rate_map_cols);
													
															auto cols = firing_rate_map_cols;
															auto rows = firing_rate_map_rows / stimulus_size;
										

															TRN4CPP::Simulation::Loop::SpatialFilter::configure(id, batch_size, stimulus_size, seed, rows, cols, x, y, firing_rate_map, sigma, radius, scale, tag);
														}
													}
													/// SCHEDULER
													else if (boost::iequals(model_element.first, scheduler_name))
													{
														auto _scheduler = model_element.second;
														auto scheduler_type = _scheduler.get_child(type_attribute).get_value<std::string>();
														if (boost::iequals(scheduler_type, tiled_type))
														{
															auto epochs = get_variable<unsigned int>(_scheduler, epochs_attribute, condition_number, simulation_number);
															TRN4CPP::Simulation::Scheduler::Tiled::configure(id, epochs);
														}
														else if (boost::iequals(scheduler_type, snippets_type))
														{
															auto seed = simulation_number + get_variable<unsigned long>(_scheduler, seed_attribute, condition_number, simulation_number);
															auto snippets_size = get_variable<std::size_t>(_scheduler, snippets_size_attribute, condition_number, simulation_number);
															auto time_budget = get_variable<std::size_t>(_scheduler, time_budget_attribute, condition_number, simulation_number);
															auto tag = _scheduler.get(tag_attribute, DEFAULT_TAG);
															TRN4CPP::Simulation::Scheduler::Snippets::configure(id, seed, snippets_size, time_budget, tag);
														}
														else if (boost::iequals(scheduler_type, custom_type))
														{
															auto seed = simulation_number + get_variable<unsigned long>(_scheduler, seed_attribute, condition_number, simulation_number);
															auto tag = _scheduler.get_child(seed_attribute).get_value<std::string>(DEFAULT_TAG);
															TRN4CPP::Simulation::Scheduler::Custom::configure(id, seed, tag);
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
																	auto seed = simulation_number + get_variable<unsigned long>(_mutator, seed_attribute, condition_number, simulation_number);
																	auto rate = get_variable<float>(_mutator, rate_attribute, condition_number, simulation_number);
																	auto size = get_variable<std::size_t>(_mutator, size_attribute, condition_number, simulation_number);
																	//auto number = _mutator.get_child(number_attribute).get_value<std::size_t>();
																	TRN4CPP::Simulation::Scheduler::Mutator::Reverse::configure(id, seed, rate, size);
																	//
																}
																else if (boost::iequals(mutator_type, punch_type))
																{
																	auto seed = simulation_number + get_variable<unsigned long>(_mutator, seed_attribute, condition_number, simulation_number);
																	auto rate = get_variable<float>(_mutator, rate_attribute, condition_number, simulation_number);
																	auto size = get_variable<std::size_t>(_mutator, size_attribute, condition_number, simulation_number);
																	auto number = get_variable<std::size_t>(_mutator, number_attribute, condition_number, simulation_number);
																	TRN4CPP::Simulation::Scheduler::Mutator::Punch::configure(id, seed, rate, size, number);
																	//
																}
																else if (boost::iequals(mutator_type, shuffle_type))
																{
																	auto seed = simulation_number + get_variable<unsigned long>(_mutator, seed_attribute, condition_number, simulation_number);

																	TRN4CPP::Simulation::Scheduler::Mutator::Shuffle::configure(id, seed);
																	//
																}
																else if (boost::iequals(mutator_type, custom_type))
																{
																	auto seed = simulation_number + get_variable<unsigned long>(_mutator, seed_attribute, condition_number, simulation_number);

																	TRN4CPP::Simulation::Scheduler::Mutator::Custom::configure(id, seed);
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
																TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::configure(id, batch_size);
																readout_mean_square_error_initialized = true;
															}
															else if (boost::iequals(measurement_type, frechet_distance_type))
															{
																TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::configure(id, batch_size);
																readout_frechet_distance_initialized = true;
															}
															else if (boost::iequals(measurement_type, custom_type))
															{
																TRN4CPP::Simulation::Measurement::Readout::Custom::configure(id, batch_size);
																readout_custom_initialized = true;
															}
														}
														else if (boost::iequals(measurement_target, position_name))
														{
															auto measurement_type = _measurement.get_child(type_attribute).get_value<std::string>();
															if (boost::iequals(measurement_type, mean_square_error_type))
															{
																TRN4CPP::Simulation::Measurement::Position::MeanSquareError::configure(id, batch_size);
																position_mean_square_error_initialized = true;
															}
															else if (boost::iequals(measurement_type, frechet_distance_type))
															{
																TRN4CPP::Simulation::Measurement::Position::FrechetDistance::configure(id, batch_size);
																position_frechet_distance_initialized = true;
															}
															else if (boost::iequals(measurement_type, custom_type))
															{
																TRN4CPP::Simulation::Measurement::Position::Custom::configure(id, batch_size);
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

															TRN4CPP::Simulation::Recording::States::configure(id, train, prime, generate);
														}
														else if (boost::iequals(measurement_target, weights_type))
														{
															auto train = _measurement.get_child(train_attribute).get_value<bool>();
															auto initialization = _measurement.get_child(initialize_attribute).get_value<bool>();

															TRN4CPP::Simulation::Recording::Weights::configure(id, initialization, train);
														}
														else if (boost::iequals(measurement_target, performances_type))
														{
															auto train = _measurement.get_child(train_attribute).get_value<bool>();
															auto prime = _measurement.get_child(prime_attribute).get_value<bool>();
															auto generate = _measurement.get_child(generate_attribute).get_value<bool>();

															TRN4CPP::Simulation::Recording::Performances::configure(id, train, prime, generate);
														}
														else if (boost::iequals(measurement_target, scheduling_type))
														{
															TRN4CPP::Simulation::Recording::Scheduling::configure(id);
														}
													}
												}

											}
										}

										TRN4CPP::Simulation::configure_end(id);

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
										std::size_t trial_number = 1;
										for (auto execution_element : _execution)
										{
							
											if (boost::iequals(execution_element.first, trial_name))
											{
												auto _trial = execution_element.second;
												bool trained = false;
												std::size_t test_number = 1;
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
										
														auto reset_readout = get_attribute(_train, reset_attribute, false);
														
														TRN4CPP::Simulation::train(id, label, incoming, expected, reset_readout);

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
														auto autonomous = _test.get_child(autonomous_attribute).get_value<bool>();
														TRN4CPP::Search::update(condition_number, simulation_number, trial_number, test_number, repeat);
														for (std::size_t k = 0; k < repeat; k++)
														{
															TRN4CPP::Simulation::test(id, label, incoming, expected, preamble, autonomous, supplementary);
														}

														test_number++;
													}
												}
												trial_number++;
											}
										}

									}
								}
								TRN4CPP::Simulation::deallocate(id);
								simulation_number++;
							}
						} 
						while (!TRN4CPP::Search::end(condition_number, simulation_number));
						condition_number++;
					}
				}
				experiment_done = true;
			}
		}
		TRN4CPP::Engine::Execution::run();
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what() ;
	}
}