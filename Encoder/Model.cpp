#include "stdafx.h"
#include "Model_impl.h"
#include "Core/Encoder_impl.h"


TRN::Encoder::Model::Model(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::shared_ptr<TRN::Core::Matrix> &cx,
	const std::shared_ptr<TRN::Core::Matrix> &cy,
	const std::shared_ptr<TRN::Core::Matrix> &width) :
	TRN::Core::Encoder(driver, batch_size, stimulus_size),
	handle(std::make_unique<Handle>())
{
	handle->cx = cx;
	handle->cy = cy;
	handle->width = width;
}

TRN::Encoder::Model::~Model()
{
	handle.reset();
}
		
void TRN::Encoder::Model::encode(
	const std::shared_ptr<TRN::Core::Batch> &decoded_position,
	const unsigned long long &evaluation_id,
	std::shared_ptr<TRN::Core::Batch> &encoded_activations)
{
	implementor->get_algorithm()->encode_placecells_model
	(
		TRN::Core::Encoder::handle->batch_size, TRN::Core::Encoder::handle->stimulus_size,
		handle->cx->get_elements(), handle->cy->get_elements(), handle->width->get_elements(),
		(const float **)decoded_position->get_elements(), decoded_position->get_strides(),
		encoded_activations->get_elements(), encoded_activations->get_strides()
	);
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::POSITION>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::POSITION>(decoded_position, evaluation_id));
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>(encoded_activations, evaluation_id));
}

std::shared_ptr<TRN::Encoder::Model> TRN::Encoder::Model::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::shared_ptr<TRN::Core::Matrix> &cx,
	const std::shared_ptr<TRN::Core::Matrix> &cy,
	const std::shared_ptr<TRN::Core::Matrix> &width)
{
	return std::make_shared<Model>(driver, batch_size, stimulus_size, cx, cy, width);
}

