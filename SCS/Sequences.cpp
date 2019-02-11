#include "stdafx.h"
#include "Sequences.h"
#include "Helper/Logger.h"
#include "TRN4JAVA/Convert.h"
#include "TRN4SCS_Sequences.h"

struct Matrix
{
	std::size_t rows;
	std::size_t cols;
	std::vector<float> elements;
};

class Sequences::Handle
{
public:
	std::function<void(const std::string &label, const std::string &tag, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)> declare;
};


static std::map<std::pair<std::string, std::string>, Matrix> storage;
static std::mutex mutex;

void Sequences::initialize(const std::map<std::string, std::string> &arguments)
{
	handle = std::make_unique<Handle>();
}

void Sequences::uninitialize()
{
	handle.reset();
}


void Java_TRN4SCS_Sequences_declare(JNIEnv *env, jclass klass, jstring label, jstring tag, jfloatArray elements, jlong observations)
{
	auto _label = TRN4JAVA::Convert::to_string(env, label);
	auto _tag = TRN4JAVA::Convert::to_string(env, tag);
	auto _elements = TRN4JAVA::Convert::to_float_vector(env, elements);
	auto _rows = static_cast<std::size_t>(observations);
	auto _cols = _elements.size() / _rows;
	auto key = std::make_pair(_label, _tag);

	std::unique_lock<std::mutex> lock(mutex);
	storage[key].rows = _rows;
	storage[key].cols = _cols;
	storage[key].elements = _elements;
}

void  Sequences::callback_variable(const std::string &label, const std::string &tag)
{
	
	auto key = std::make_pair(label, tag);
	std::unique_lock<std::mutex> lock(mutex);
	auto &matrix = storage[key];
	handle->declare(label, tag, matrix.elements, matrix.rows, matrix.cols);
}
void  Sequences::install_variable(const std::function<void(const std::string &label, const std::string &tag, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	handle->declare = functor;
}
