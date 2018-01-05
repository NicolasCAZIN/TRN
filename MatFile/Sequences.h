#pragma once

#include "TRN4CPP/Sequences.h"

class Sequences :  public TRN4CPP::Plugin::Sequences::Interface
{
private:
	class Handle;
	std::unique_ptr<Handle> handle;

public :
	virtual void initialize(const std::map<std::string, std::string> &arguments) override;
	virtual void uninitialize() override;
public :
	virtual void callback_variable(const std::string &label, const std::string &tag) override;
	virtual void install_variable(const std::function<void(const std::string &label, const std::string &tag, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)> &functor) override;
};
extern "C" BOOST_SYMBOL_EXPORT Sequences plugin_sequences;
Sequences plugin_sequences;



