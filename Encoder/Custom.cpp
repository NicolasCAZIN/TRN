#include "stdafx.h"
#include "Custom_impl.h"
#include "Core/Encoder_impl.h"

TRN::Encoder::Custom::Custom(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus) :
	TRN::Core::Encoder(driver, batch_size, stimulus_size),
	handle(std::make_unique<Handle>())
{
	handle->on_predicted_position = predicted_position;
	handle->stimulus = TRN::Core::Batch::create(driver, batch_size);
	handle->position = TRN::Core::Batch::create(driver, batch_size);
	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		handle->stimulus->update(batch, TRN::Core::Matrix::create(driver, 1, stimulus_size));
		handle->position->update(batch, TRN::Core::Matrix::create(driver, 1, 2));
	}
	estimated_position = [this](const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
	{
		assert(rows == TRN::Core::Encoder::handle->batch_size);
		for (std::size_t batch = 0; batch < TRN::Core::Encoder::handle->batch_size; batch++)
		{
			std::vector<float> local(position.begin() + batch * cols, position.begin() + batch * cols + cols);
			assert(local.size() == cols);
			handle->position->get_matrices(batch)->from(local, 1, cols);
		}
		TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::POSITION>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::POSITION>(handle->position, evaluation_id));
	};

	perceived_stimulus = [this](const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
	{
		assert(rows == TRN::Core::Encoder::handle->batch_size);
		for (std::size_t batch = 0; batch < TRN::Core::Encoder::handle->batch_size; batch++)
		{
			std::vector<float> local(stimulus.begin() + batch * cols, stimulus.begin() + batch * cols + cols);
			assert(local.size() == cols);
			handle->stimulus->get_matrices(batch)->from(local, 1, cols);
		}
		TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>(handle->stimulus, evaluation_id));
	};
}

TRN::Encoder::Custom::~Custom()
{
	handle.reset();
}

void TRN::Encoder::Custom::encode(
	const std::shared_ptr<TRN::Core::Batch> &decoded_position,
	const unsigned long long &evaluation_id,
	std::shared_ptr<TRN::Core::Batch> &encoded_activations)
{
	std::vector<float> position;
	std::vector<std::size_t> rows, cols;
	std::size_t batch_size;

	decoded_position->to(position, batch_size, rows, cols);
	handle->on_predicted_position(evaluation_id, position, batch_size, cols[0]);
}

std::shared_ptr<TRN::Encoder::Custom> TRN::Encoder::Custom::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus)
{
	return std::make_shared<TRN::Encoder::Custom>(driver, batch_size, stimulus_size, predicted_position, estimated_position, perceived_stimulus);
}