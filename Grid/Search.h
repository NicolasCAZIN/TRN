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

	virtual void callback_generation(const unsigned short &condition_number, const std::vector<float> &score) override;
	virtual void install_generation(const std::function<void(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population)> &functor) override;
};
extern "C" BOOST_SYMBOL_EXPORT Search plugin_search;
Search plugin_search;
