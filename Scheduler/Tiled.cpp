#include "stdafx.h"
#include "Tiled.h"

class TRN::Scheduler::Tiled::Handle
{
public :
	mutable int epochs;
};

TRN::Scheduler::Tiled::Tiled(const unsigned int &epochs) :
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

	auto incoming = delegate.lock()->retrieve_set(payload.get_label(), payload.get_incoming());
	/*std::vector<float> seq_elements;
	std::size_t seq_rows, seq_cols;
	incoming->get_sequence()->to(seq_elements, seq_rows, seq_cols);

	cv::Mat seq(seq_rows, seq_cols, CV_32F, seq_elements.data());*/
	std::vector<int> offsets(handle->epochs * incoming->get_sequence()->get_rows());
	std::vector<int> durations(handle->epochs);

	std::size_t k = 0;
	for (std::size_t epoch = 0; epoch < handle->epochs; epoch++)
	{
		for (std::size_t t = 0; t < incoming->get_sequence()->get_rows(); t++)
		{
			offsets[k] = t;
			k++;
		}
		durations[epoch] = incoming->get_sequence()->get_rows();
	}

	notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(payload.get_evaluation_id(), TRN::Core::Scheduling::create(offsets, durations)));
}


std::shared_ptr<TRN::Scheduler::Tiled> TRN::Scheduler::Tiled::create( const unsigned int &epochs)
{
	return std::make_shared<TRN::Scheduler::Tiled>( epochs);
}