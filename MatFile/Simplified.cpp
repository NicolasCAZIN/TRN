#include "stdafx.h"
#include "Simplified.h"

const std::string FILENAME_TOKEN = "FILENAME";
const std::string MAPPING_TOKEN = "MAPPING";
const std::string VARIABLE_TOKEN = "VARIABLE";

struct Simplified::Handle 
{
	MATFile *pmat;
	std::map<std::pair<std::string, std::string>, std::vector<std::string>> fields;
	std::function<void(const std::string &label, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::string &tag)> declare;
};

void Simplified::initialize(const std::map<std::string, std::string> &arguments)
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

	for (auto properties_element : properties)
	{
		if (boost::iequals(properties_element.first, VARIABLE_TOKEN))
		{
			auto _variable = properties_element.second;

			auto label = _variable.get_child(LABEL_ATTRIBUTE).get_value<std::string>();
			auto tag = _variable.get(TAG_ATTRIBUTE, "");

			std::vector<std::string> path;
			boost::split(path, _variable.get_child(PATH_ATTRIBUTE).get_value<std::string>(), boost::is_any_of("."));
			handle->fields[std::make_pair(label, tag)] = path;
		}
	}

	if (handle->pmat != NULL)
		throw std::runtime_error("a MAT-File is already opened");
	handle->pmat = matOpen(filename.c_str(), "r");
	if (handle->pmat == NULL)
		throw std::runtime_error("Can't open MAT file " + filename);
}
void Simplified::uninitialize()
{
	if (matClose(handle->pmat))
		throw std::runtime_error("Can't close matfile handle");
	handle.reset();
}
void  Simplified::callback_variable(const std::string &label, const std::string &tag)
{
	if (!handle->declare)
		throw std::runtime_error("Declare functor is not installed");

	auto key = std::make_pair(label, tag);
	if (handle->fields.find(key) == handle->fields.end())
		throw std::runtime_error("variable with label "  +label + " and tag "+ tag + "is not mapped");
	auto path = handle->fields[key];

	auto mx_variable = matGetVariable(handle->pmat, path[0].c_str());
	if (mx_variable == NULL)
		throw std::runtime_error("Can't read variable " + label);

	try
	{
		auto mx_field = mx_variable;
		std::size_t depth = 1;
		while (mxIsStruct(mx_variable) && depth < path.size())
		{
			mx_field = mxGetField(mx_field, 0, path[depth].c_str());
			depth++;
		}
		if (mxIsNumeric(mx_field))
		{
			if (mxIsSingle(mx_field))
			{
				auto dims = mxGetDimensions(mx_field);
				if (mxGetNumberOfDimensions(mx_field) == 2)
				{
					auto rows = dims[1];
					auto cols = dims[0];
					auto size = rows * cols;
					float *ptr = (float *)mxGetData(mx_field);
					std::vector<float> elements(ptr, ptr + size);
					handle->declare(label, elements, rows, cols, tag);
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
		mxDestroyArray(mx_variable);
	}
	catch (std::exception &e)
	{
		mxDestroyArray(mx_variable);
		throw e;
	}
}
void  Simplified::install_variable(const std::function<void(const std::string &label, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::string &tag)> &functor)
{
	handle->declare = functor;
}

/*boost::shared_ptr<Simplified>  Simplified::create()
{
	return boost::make_shared<Simplified>();
}*/