#include "stdafx.h"
#include "Measurement_impl.h"

TRN::Core::Measurement::Implementation::Implementation(const std::shared_ptr<TRN::Backend::Driver> &driver) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver)
{

}

TRN::Core::Measurement::Implementation::~Implementation() noexcept(false)
{

}

TRN::Core::Measurement::Abstraction::Abstraction(const std::shared_ptr<TRN::Core::Measurement::Implementation> &compute, const std::size_t &batch_size):
	TRN::Helper::Bridge<TRN::Core::Measurement::Implementation>(compute),
	handle(std::make_unique<Handle>())
{
	handle->stored = 0;
	handle->preamble = 0;
	handle->measurable_generations = 0;
	handle->expected_generations = 0;
	handle->supplementary_generations = 0;
	handle->batch_size = batch_size;
	handle->batched_predicted = TRN::Core::Batch::create(implementor->get_implementor(), batch_size);
}

TRN::Core::Measurement::Abstraction::~Abstraction() noexcept(false)
{
	handle.reset();
}

void TRN::Core::Measurement::Abstraction::update(const TRN::Core::Message::Payload<TRN::Core::Message::TEST> &payload)
{
	handle->preamble = payload.get_preamble();
	handle->supplementary_generations = payload.get_supplementary_generations();
}

void TRN::Core::Measurement::Abstraction::set_expected(const std::shared_ptr<TRN::Core::Matrix> &expected)
{
	
	auto cols = expected->get_cols();
	handle->measurable_generations = expected->get_rows() - handle->preamble;
	handle->expected_generations = handle->measurable_generations + handle->supplementary_generations;
	handle->primed = TRN::Core::Matrix::create(implementor->get_implementor(), expected, 0, 0, handle->preamble, cols);

	auto sub = TRN::Core::Matrix::create(implementor->get_implementor(), expected, handle->preamble, 0, handle->measurable_generations, cols);
	handle->expected = TRN::Core::Matrix::create(implementor->get_implementor(), handle->measurable_generations, cols, true);
	handle->expected->from(*sub);

	handle->error = TRN::Core::Matrix::create(implementor->get_implementor(), 1, handle->batch_size);

	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
	{
		handle->batched_predicted->update(batch, TRN::Core::Matrix::create(implementor->get_implementor(), handle->expected_generations, cols));
	}
	handle->stored = 0;
}

void TRN::Core::Measurement::Abstraction::on_update(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Batch> &predicted)
{
	if (handle->stored < handle->expected_generations)
	{
		for (std::size_t batch = 0; batch < handle->batch_size; batch++)
		{
			auto slice = TRN::Core::Matrix::create(implementor->get_implementor(), handle->batched_predicted->get_matrices(batch), handle->stored, 0, 1, handle->expected->get_cols());
			slice->from(*predicted->get_matrices(batch));
		}
	}

	handle->stored++;
	if (handle->stored == handle->measurable_generations)
	{
		implementor->compute(evaluation_id, handle->primed, handle->batched_predicted, handle->expected, handle->error);
	
	}
	else if (handle->stored == handle->expected_generations)
	{
		handle->stored = 0;
	}
}
