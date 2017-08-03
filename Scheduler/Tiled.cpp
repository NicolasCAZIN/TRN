#include "stdafx.h"
#include "Tiled.h"

class TRN::Scheduler::Tiled::Handle
{
public :
	mutable int epochs;
};

TRN::Scheduler::Tiled::Tiled(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &epochs) :
	TRN::Core::Scheduler(driver),
	handle(std::make_unique<TRN::Scheduler::Tiled::Handle>())
{
	handle->epochs = epochs;
}
TRN::Scheduler::Tiled::~Tiled()
{
	handle.reset();
}

void  TRN::Scheduler::Tiled::update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload)
{
	std::vector<unsigned int> offsets(handle->epochs);
	std::vector<unsigned int> durations(handle->epochs);

	auto incoming = delegate.lock()->retrieve_set(payload.get_label(), payload.get_incoming());

	std::fill(offsets.begin(), offsets.end(), 0);
	std::fill(durations.begin(), durations.end(), incoming->get_sequence()->get_rows());
	notify(TRN::Core::Scheduling::create(implementor, offsets, durations));
}


std::shared_ptr<TRN::Scheduler::Tiled> TRN::Scheduler::Tiled::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &epochs)
{
	return std::make_shared<TRN::Scheduler::Tiled>(driver, epochs);
}