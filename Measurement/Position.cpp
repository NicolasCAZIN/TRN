#include "stdafx.h"
#include "Position.h"
#include "Core/Measurement_impl.h"

TRN::Measurement::Position::Position(const std::shared_ptr<TRN::Core::Measurement::Implementation> &compute, const std::size_t &batch_size) :
	TRN::Core::Measurement::Abstraction(compute, batch_size)
{

}

void TRN::Measurement::Position::update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload)
{

}
void TRN::Measurement::Position::update(const TRN::Core::Message::Payload<TRN::Core::Message::POSITION> &payload)
{
	if (!handle->expected)
		throw std::invalid_argument("Expected matrix is not set");

	on_update(payload.get_evaluation_id(), payload.get_position());
}
void TRN::Measurement::Position::update(const TRN::Core::Message::Payload<TRN::Core::Message::TARGET_SEQUENCE> &payload)
{

}
void TRN::Measurement::Position::update(const TRN::Core::Message::Payload<TRN::Core::Message::TARGET_TRAJECTORY> &payload)
{
	set_expected(payload.get_trajectory());
}

std::shared_ptr<TRN::Measurement::Position> TRN::Measurement::Position::create(const std::shared_ptr<TRN::Core::Measurement::Implementation> &compute, const std::size_t &batch_size)
{
	return std::make_shared<TRN::Measurement::Position>(compute, batch_size);
}