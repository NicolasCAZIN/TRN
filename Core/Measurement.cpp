#include "stdafx.h"
#include "Measurement_impl.h"

TRN::Core::Measurement::Implementation::Implementation(const std::shared_ptr<TRN::Backend::Driver> &driver) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver)
{

}

TRN::Core::Measurement::Implementation::~Implementation()
{

}

TRN::Core::Measurement::Abstraction::Abstraction(const std::shared_ptr<TRN::Core::Measurement::Implementation> &compute, const std::size_t &batch_size):
	TRN::Helper::Bridge<TRN::Core::Measurement::Implementation>(compute),
	handle(std::make_unique<Handle>())
{
	handle->stored = 0;
	handle->preamble = 0;
	handle->batch_size = batch_size;
	handle->batched_predicted = TRN::Core::Batch::create(implementor->get_implementor(), batch_size);
}

TRN::Core::Measurement::Abstraction::~Abstraction()
{
	handle.reset();
}

void TRN::Core::Measurement::Abstraction::update(const TRN::Core::Message::Payload<TRN::Core::Message::TEST> &payload)
{
	handle->preamble = payload.get_preamble();
}

void TRN::Core::Measurement::Abstraction::set_expected(const std::shared_ptr<TRN::Core::Matrix> &expected)
{
	auto sub = TRN::Core::Matrix::create(implementor->get_implementor(), expected, handle->preamble, 0, expected->get_rows() - handle->preamble, expected->get_cols());
	handle->expected = TRN::Core::Matrix::create(implementor->get_implementor(), sub->get_rows(), sub->get_cols());
	handle->expected->from(*sub);

	handle->error = TRN::Core::Matrix::create(implementor->get_implementor(), handle->expected->get_rows(), handle->batch_size);

	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
	{
		handle->batched_predicted->update(batch, TRN::Core::Matrix::create(implementor->get_implementor(), handle->expected->get_rows(), handle->expected->get_cols()));
	}
	handle->stored = 0;
}

void TRN::Core::Measurement::Abstraction::on_update(const std::shared_ptr<TRN::Core::Batch> &predicted)
{
	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
	{
		auto slice = TRN::Core::Matrix::create(implementor->get_implementor(), handle->batched_predicted->get_matrices(batch), handle->stored, 0, 1, handle->expected->get_cols());
		slice->from(*predicted->get_matrices(batch));
	}

	handle->stored++;
	if (handle->stored == handle->expected->get_rows())
	{
		implementor->compute( handle->batched_predicted, handle->expected, handle->error);
		handle->stored = 0;
	}
}
