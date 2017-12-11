#include "stdafx.h"
#include "Sequences.h"

#include "Helper/Logger.h"

struct Sequence
{
	std::size_t rows;
	std::size_t cols;
	std::vector<float> elements;
};

const std::string TRN4CPP::Sequences::DEFAULT_TAG = "";
static std::map<std::pair<std::string, std::string>, Sequence> sequences_map;
static boost::shared_ptr<TRN4CPP::Plugin::Sequences::Interface> sequences;
static std::recursive_mutex mutex;

void TRN4CPP::Plugin::Sequences::initialize(const std::string &library_path, const std::string &name, const std::map<std::string, std::string>  &arguments)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (sequences)
		throw std::runtime_error("A Sequences plugin is already loaded");
	boost::filesystem::path path = library_path;

	path /= name;

	sequences = boost::dll::import<TRN4CPP::Plugin::Sequences::Interface>(path, "plugin_sequences", boost::dll::load_mode::append_decorations);
	sequences->initialize(arguments);
	sequences->install_variable(std::bind(&TRN4CPP::Sequences::declare, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
	INFORMATION_LOGGER << "Variables plugin " << name << " loaded from path " << library_path;
}

void TRN4CPP::Sequences::fetch(const std::string &label, const std::string &tag)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (!sequences)
		throw std::runtime_error("Sequences plugin is not loaded");
	sequences->callback_variable(label, tag);
}

void TRN4CPP::Sequences::declare(const std::string &label, const std::string &tag, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	auto key = std::make_pair(label, tag);
	if (sequences_map.find(key) != sequences_map.end())
		throw std::invalid_argument("Sequence have already been declared");
	sequences_map[key].rows = rows;
	sequences_map[key].cols = cols;
	sequences_map[key].elements = elements;
}

void TRN4CPP::Sequences::retrieve(const std::string &label, const std::string &tag, std::vector<float> &elements, std::size_t &rows, std::size_t &cols)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	auto key = std::make_pair(label, tag);
	if (sequences_map.find(key) == sequences_map.end())
		throw std::runtime_error("Sequence having label " + label + " and tag " + tag + "does not exist");
	auto data = sequences_map[key];
	elements = data.elements;
	rows = data.rows;
	cols = data.cols;
}


void TRN4CPP::Plugin::Sequences::uninitialize()
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (sequences)
	{
		sequences->uninitialize();
		sequences.reset();
	}
}