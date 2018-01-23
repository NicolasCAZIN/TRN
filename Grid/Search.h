#pragma once

#include "grid_global.h"
#include "TRN4CPP/Search.h"

class GRID_EXPORT Search : public TRN4CPP::Plugin::Search::Interface
{
private:
	class Handle;
	std::unique_ptr<Handle> handle;

public :
	virtual void initialize(const std::map<std::string, std::string> &arguments) override;
	virtual void uninitialize() override;

	virtual void callback_generation(const unsigned short &condition_number, const std::vector< std::map < std::size_t, std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>>>>> &measurements) override;
	virtual void install_generation(const std::function<void(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population)> &functor) override;
	virtual void install_solutions(const std::function<void(const unsigned short &condition_number, const std::vector<std::pair<std::map<std::string, std::string>, float>> &solutions)> &functor) override;
private :
	float evaluate_cost(const std::map < std::size_t, std::map<std::size_t, std::pair<float, std::vector<float>>> > &measurements);
};
extern "C" BOOST_SYMBOL_EXPORT Search plugin_search;
Search plugin_search;
