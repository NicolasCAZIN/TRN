#include "stdafx.h"
#include "Custom.h"
#include "Helper/Logger.h"

class TRN::Scheduler::Custom::Handle
{
public:
	unsigned long seed;
	std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> request;
	std::string tag;
};

TRN::Scheduler::Custom::Custom(const unsigned long &seed,
	const std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed,const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply,
	const std::string &tag) :
	handle(std::make_unique<TRN::Scheduler::Custom::Handle>())
{
	handle->seed = seed;
	handle->tag = tag;
	handle->request = request;
	reply = [this](const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		try
		{
			if (offsets.size() != durations.size())
				throw std::invalid_argument("offsets and durations vectors must have the same length");
			notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(trial, TRN::Core::Scheduling::create(offsets, durations)));
		}
		catch (std::exception &e)
		{
			ERROR_LOGGER << e.what() ;
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

		handle->request(payload.get_evaluation_id(), handle->seed,  elements, rows, cols, offsets, durations);
		handle->seed += offsets.size() * durations.size();
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what() ;
	}
}

std::shared_ptr<TRN::Scheduler::Custom> TRN::Scheduler::Custom::create(const unsigned long &seed,
	const std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply,
	const std::string &tag)
{
	return std::make_shared<TRN::Scheduler::Custom>(seed, request, reply, tag);
}
