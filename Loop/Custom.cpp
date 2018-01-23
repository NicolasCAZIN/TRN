#include "stdafx.h"
#include "Custom.h"
#include "Core/Loop_impl.h"
#include <iostream>

class TRN::Loop::Custom::Handle
{
public :
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> request;
};

TRN::Loop::Custom::Custom(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply
	) :
	TRN::Core::Loop(driver, batch_size, stimulus_size),
	handle(std::make_unique<Handle>())
{
	handle->request = request;
	reply = [this](const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
	{
		assert(rows == TRN::Core::Loop::handle->batch_size);
		for (std::size_t batch = 0; batch < TRN::Core::Loop::handle->batch_size; batch++)
		{
			std::vector<float> local(stimulus.begin() + batch * cols, stimulus.begin() + batch * cols + cols - 1);
			assert(local.size() == cols);
			TRN::Core::Loop::handle->stimulus->get_matrices(batch)->from(local, 1, cols);
		}
	
		TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>(TRN::Core::Loop::handle->stimulus, evaluation_id));
	};
}

TRN::Loop::Custom::~Custom()
{
	handle.reset();
}

void TRN::Loop::Custom::update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload)
{
	auto predicted = payload.get_predicted();
	std::vector<float> action;
	std::size_t  matrices;
	std::vector<std::size_t> rows;
	std::vector<std::size_t> cols;

	assert(std::all_of(rows.begin(), rows.end(), [&](const std::size_t &r) { return r == 1; }));
	assert(std::all_of(cols.begin(), cols.end(), [&](const std::size_t &c) { return c == cols[0]; }));

	predicted->to(action, matrices, rows, cols);
	handle->request(payload.get_evaluation_id(), action, matrices, cols[0]);
}

void TRN::Loop::Custom::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload)
{
	payload->set_flops_per_cycle(0);
	payload->set_flops_per_epoch_factor(0);
}

std::shared_ptr<TRN::Loop::Custom> TRN::Loop::Custom::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &prediction,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &stimulus)
{
	return std::make_shared<TRN::Loop::Custom>(driver, batch_size, stimulus_size, prediction, stimulus);
}