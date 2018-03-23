#include "stdafx.h"
#include "Decoder_impl.h"


TRN::Core::Decoder::Decoder(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size) : 
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<Handle>())
{
	handle->batch_size = batch_size;
	handle->stimulus_size = stimulus_size;
}
TRN::Core::Decoder::~Decoder()
{
	handle.reset();
}