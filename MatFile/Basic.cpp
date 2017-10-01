#include "stdafx.h"
#include "Basic.h"

const std::string FILENAME_TOKEN = "FILENAME";
const std::string MAPPING_TOKEN = "MAPPING";
const std::string VARIABLE_TOKEN = "VARIABLE";

Basic::Basic()
{
	pmat = NULL;
}

Basic::~Basic()
{
	uninitialize();
}

void Basic::initialize(const std::string &filename, const std::string &mode)
{
	if (pmat != NULL)
		throw std::runtime_error("a MAT-File is already opened");
	pmat = matOpen(filename.c_str(), mode.c_str());
	if (pmat == NULL)
		throw std::runtime_error("Can't open MAT file " + filename);
}

void Basic::uninitialize()
{
	if (pmat != NULL)
	{
		if (matClose(pmat) != 0)
			throw std::runtime_error("Can't close MAT file");
		pmat = NULL;
	}
}
