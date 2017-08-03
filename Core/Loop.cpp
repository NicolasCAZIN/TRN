#include "stdafx.h"
#include "Loop_impl.h"

TRN::Core::Loop::Loop(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<Handle>())
{
	handle->batch_size = batch_size;
	handle->stimulus = TRN::Core::Batch::create(driver, batch_size);
	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		handle->stimulus->update(batch, TRN::Core::Matrix::create(driver, 1, stimulus_size));
	}
}

TRN::Core::Loop::~Loop()
{
	implementor->synchronize();
	handle.reset();
}
void TRN::Core::Loop::update(const TRN::Core::Message::Payload<TRN::Core::Message::TEST> &payload)
{
	
	// do nothing
}
