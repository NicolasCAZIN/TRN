#pragma once

#include "simulatedannealing_global.h"
#include "TRN4CPP/Search.h"

class SIMULATEDANNEALING_EXPORT Search : public TRN4CPP::Plugin::Search::Interface
{
private:
	class Handle;
	std::unique_ptr<Handle> handle;

public:
	virtual void initialize(const std::map<std::string, std::string> &arguments) override;
	virtual void uninitialize() override;

	virtual void callback_generation(const unsigned short &condition_number, const std::vector<std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>> &measurements) override;
	virtual void install_generation(const std::function<void(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population)> &functor) override;
	virtual void install_solutions(const std::function<void(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions)> &functor) override;

private :
	void reset();
	std::size_t select(const std::size_t &size);
	std::string choose(const std::vector<std::string> &possible);
	std::map<std::string, std::string> neighboor(const std::map<std::string, std::string> &candidate);
};
extern "C" BOOST_SYMBOL_EXPORT Search plugin_search;
Search plugin_search;
