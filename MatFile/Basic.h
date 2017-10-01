#pragma once

#include "matfile_global.h"
#include "TRN4CPP/Basic.h"

extern const std::string FILENAME_TOKEN;
extern const std::string MAPPING_TOKEN;
extern const std::string VARIABLE_TOKEN;

class MATFILE_EXPORT Basic : public TRN4CPP::Plugin::Basic::Interface
{
protected:
	MATFile *pmat;
protected :
	Basic();
	virtual ~Basic();

protected:
	void initialize(const std::string &filename, const std::string &mode);

public:
	virtual void uninitialize() override;
};
