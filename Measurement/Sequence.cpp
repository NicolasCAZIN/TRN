#include "stdafx.h"
#include "Sequence.h"
#include "Core/Measurement_impl.h"

TRN::Measurement::Sequence::Sequence(const std::shared_ptr<TRN::Core::Measurement::Implementation> &compute, const std::size_t &batch_size) :
	TRN::Core::Measurement::Abstraction(compute, batch_size)
{

}

void TRN::Measurement::Sequence::update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload)
{
	if (!handle->expected)
		throw std::invalid_argument("Expected matrix is not set");

	on_update(payload.get_evaluation_id(), payload.get_predicted());
}
void TRN::Measurement::Sequence::update(const TRN::Core::Message::Payload<TRN::Core::Message::POSITION> &payload)
{

}
void TRN::Measurement::Sequence::update(const TRN::Core::Message::Payload<TRN::Core::Message::TARGET_SEQUENCE> &payload)
{
	set_expected(payload.get_sequence());
}
void TRN::Measurement::Sequence::update(const TRN::Core::Message::Payload<TRN::Core::Message::TARGET_TRAJECTORY> &payload)
{
	
}

std::shared_ptr<TRN::Measurement::Sequence> TRN::Measurement::Sequence::create(const std::shared_ptr<TRN::Core::Measurement::Implementation> &compute, const std::size_t &batch_size)
{
	return std::make_shared<TRN::Measurement::Sequence>(compute, batch_size);
}