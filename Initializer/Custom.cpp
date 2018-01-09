#include "stdafx.h"
#include "Custom.h"
#include "Helper/Logger.h"

class TRN::Initializer::Custom::Handle
{
public:
	/*std::mutex mutex;
	std::condition_variable cond;*/
	std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> request;
	std::queue<std::pair<std::shared_ptr<TRN::Core::Batch>,bool>> pending;
};

TRN::Initializer::Custom::Custom(const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply) :
	TRN::Core::Initializer(driver),
	handle(std::make_unique<TRN::Initializer::Custom::Handle>())
{
	handle->request = request;
	reply = [this](const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		try
		{
			auto batch = handle->pending.front().first;
			auto blank_diagonal = handle->pending.front().second;
			if (matrices != batch->get_size())
				throw std::invalid_argument("invalid number of matrix in batch");
			auto wb = weights.begin();
		
			for (std::size_t matrix = 0; matrix < matrices; matrix++)
			{
				auto we = wb + rows*cols;
				std::vector<float> sub_weights(wb, we);
				if (blank_diagonal && rows == cols)
				{
					for (int row = 0; row < rows; row++)
						sub_weights[row * cols + row] = 0.0f;
				}
				batch->get_matrices(matrix)->from(sub_weights, rows, cols);
				wb = we;
			}

			handle->pending.pop();
		}
		catch (std::exception &e)
		{
			ERROR_LOGGER << e.what() ;
		}
	};
}
TRN::Initializer::Custom::~Custom()
{
	handle.reset();
}

void TRN::Initializer::Custom::initialize(unsigned long &seed, std::shared_ptr<TRN::Core::Batch> &batch, const bool &blank_diagonal)
{
	handle->pending.push(std::make_pair(batch, blank_diagonal));
	handle->request(seed, batch->get_size(), *batch->get_rows(), *batch->get_cols());
	seed += batch->get_size() * *batch->get_rows() * *batch->get_cols();
}

std::shared_ptr<TRN::Initializer::Custom> TRN::Initializer::Custom::create(const std::shared_ptr<TRN::Backend::Driver> &driver, 
	const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	return std::make_shared<TRN::Initializer::Custom>(driver, request, reply);
}
