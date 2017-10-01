#pragma once

#include "Basic.h"
#include "TRN4CPP/Simplified.h"
#include <boost/dll/alias.hpp> 

class Simplified :  public TRN4CPP::Plugin::Simplified::Interface, public Basic

{
private:
	std::map<std::pair<std::string, std::string>, std::vector<std::string>> fields;
	std::function<void(const std::string &label, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::string &tag)> declare;

public :
	virtual void initialize(const std::map<std::string, std::string> &arguments) override;
	virtual void uninitialize() override;
public :
	virtual void callback_variable(const std::string &label, const std::string &tag) override;
	virtual void install_variable(const std::function<void(const std::string &label, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::string &tag)> &functor) override;

	
};
extern "C" BOOST_SYMBOL_EXPORT Simplified plugin_simplified;
Simplified plugin_simplified;



