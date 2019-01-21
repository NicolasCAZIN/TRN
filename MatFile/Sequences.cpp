#include "stdafx.h"
#include "Sequences.h"
#include "Helper/Logger.h"

const std::string FILENAME_TOKEN = "FILENAME";
const std::string MAPPING_TOKEN = "MAPPING";
const std::string VARIABLE_TOKEN = "VARIABLE";

struct Matrix
{
	std::size_t rows;
	std::size_t cols;
	std::vector<float> elements;
};

struct Sequences::Handle
{
	std::map<std::pair<std::string, std::string>, Matrix> fields;
	std::function<void(const std::string &label, const std::string &tag, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)> declare;
};


void Sequences::initialize(const std::map<std::string, std::string> &arguments)
{
	if (handle)
		throw std::runtime_error("Handle already allocated");
	handle = std::make_unique<Handle>();
	std::string prefix = "";

	if (arguments.find(FILENAME_TOKEN) == arguments.end())
		throw std::runtime_error("Can't file " + FILENAME_TOKEN + " key/value pair");
	auto filename = arguments.at(FILENAME_TOKEN);

	if (arguments.find(MAPPING_TOKEN) == arguments.end())
		throw std::runtime_error("Can't file " + MAPPING_TOKEN + " key/value pair");
	auto mapping = arguments.at(MAPPING_TOKEN);


	boost::property_tree::iptree properties;
	auto extension = boost::filesystem::extension(mapping);
	if (boost::iequals(extension, ".INI"))
		boost::property_tree::read_ini(mapping, properties);
	else if (boost::iequals(extension, ".XML"))
	{
		boost::property_tree::read_xml(mapping, properties);
		prefix = "<xmlattr>.";
	}
	else if (boost::iequals(extension, ".JSON"))
		boost::property_tree::read_json(mapping, properties);
	else if (boost::iequals(extension, ".INFO"))
		boost::property_tree::read_info(mapping, properties);
	else
		throw std::runtime_error("Unexpected file extension");

	const std::string PATH_ATTRIBUTE = prefix + "path";
	const std::string LABEL_ATTRIBUTE = prefix + "label";
	const std::string TAG_ATTRIBUTE = prefix + "tag";

	MATFile *pmat = matOpen(filename.c_str(), "r");

	if (pmat == NULL)
		throw std::runtime_error("Can't open MAT file " + filename);

	int num = 0;
	auto dirs = matGetDir(pmat, &num);
	std::map<std::string, mxArray *> variables;
	for (int n = 0; n < num; n++)
	{
		auto dir = dirs[n];
		auto mx_variable = matGetVariable(pmat, dir);
		variables[dir] = mx_variable;

	}

	mxFree(dirs);

	for (auto properties_element : properties)
	{
		if (boost::iequals(properties_element.first, MAPPING_TOKEN))
		{
			auto _mapping = properties_element.second;

			for (auto mapping_element : _mapping)
			{
				if (boost::iequals(mapping_element.first, VARIABLE_TOKEN))
				{
					auto _variable = mapping_element.second;

					auto label = _variable.get_child(LABEL_ATTRIBUTE).get_value<std::string>();
					auto tag = _variable.get(TAG_ATTRIBUTE, "");

					std::vector<std::string> path;
					boost::split(path, _variable.get_child(PATH_ATTRIBUTE).get_value<std::string>(), boost::is_any_of("."));
					auto key = std::make_pair(label, tag);
					if (variables.find(path[0]) == variables.end())
					{
						throw std::runtime_error("Variable " + path[0] + " does not exist");
					}
					auto mx_variable = variables[path[0]];
					auto mx_field = mx_variable;
					std::size_t depth = 1;
					while (mx_field && mxIsStruct(mx_field) && depth < path.size())
					{
						mx_field = mxGetField(mx_field, 0, path[depth].c_str());
						depth++;
					}
					if (mx_field)
					{
						if (mxIsNumeric(mx_field))
						{
							if (mxIsSingle(mx_field))
							{
								auto dims = mxGetDimensions(mx_field);
								if (mxGetNumberOfDimensions(mx_field) == 2)
								{
									handle->fields[key].rows = dims[1];
									handle->fields[key].cols = dims[0];
									auto size = dims[1] * dims[0];
									float *ptr = (float *)mxGetData(mx_field);
									handle->fields[key].elements.resize(size);
									std::copy(ptr, ptr + size, handle->fields[key].elements.begin());
								}
								else
								{
									throw std::runtime_error("variable " + label + " is not numeric");
								}
							}
							else
							{
								throw std::runtime_error("variable " + label + " is not single precision");
							}
						}
					}
					else
					{
						throw std::runtime_error("variable " + label + " does not exist");
					}
				}
			}
		}

	}
	for (auto variable : variables)
	{
		mxDestroyArray(variable.second);
	}
	
	matClose(pmat);



	

}
void Sequences::uninitialize()
{
	handle.reset();
}
void  Sequences::callback_variable(const std::string &label, const std::string &tag)
{
	if (!handle->declare)
		throw std::runtime_error("Declare functor is not installed");

	auto key = std::make_pair(label, tag);
	if (handle->fields.find(key) == handle->fields.end())
		throw std::runtime_error("variable with label "  +label + " and tag "+ tag + "is not mapped");
	auto &matrix = handle->fields[key];

	handle->declare(label, tag, matrix.elements, matrix.rows, matrix.cols);
}
void  Sequences::install_variable(const std::function<void(const std::string &label, const std::string &tag, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	handle->declare = functor;
}
