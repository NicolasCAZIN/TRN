#include "stdafx.h"
#include "Custom.h"

class TRN::Scheduler::Custom::Handle
{
public:
	std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> request;
	std::string tag;
};

TRN::Scheduler::Custom::Custom(const std::shared_ptr<TRN::Backend::Driver> &driver, 
	const std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &reply,
	const std::string &tag) :
	TRN::Core::Scheduler(driver),
	handle(std::make_unique<TRN::Scheduler::Custom::Handle>())
{
	handle->tag = tag;
	handle->request = request;
	reply = [this](const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		try
		{
			if (offsets.size() != durations.size())
				throw std::invalid_argument("offsets and durations vectors must have the same length");
			notify(TRN::Core::Scheduling::create(offsets, durations));
		}
		catch (std::exception &e)
		{
			std::cerr << e.what() << std::endl;
		}
	};
}

TRN::Scheduler::Custom::~Custom()
{
	handle.reset();
}

void TRN::Scheduler::Custom::update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload)
{
	try
	{
		std::vector<int> offsets;
		std::vector<int> durations;
		std::size_t rows;
		std::size_t cols;
		std::vector<float> elements;
		auto batch = delegate.lock()->retrieve_set(payload.get_label(), handle->tag);
		batch->get_sequence()->to(elements, rows, cols);
		batch->get_scheduling()->to(offsets, durations);

		handle->request(elements, rows, cols, offsets, durations);
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

std::shared_ptr<TRN::Scheduler::Custom> TRN::Scheduler::Custom::create(const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &reply,
	const std::string &tag)
{
	return std::make_shared<TRN::Scheduler::Custom>(driver, request, reply, tag);
}
