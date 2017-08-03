#pragma once

#include "Message.h"

class TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>::Handle
{
public:
	std::shared_ptr<TRN::Core::Batch> stimulus;
};

class TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION>::Handle
{
public:
	std::shared_ptr<TRN::Core::Batch> predicted;
};
class TRN::Core::Message::Payload<TRN::Core::Message::POSITION>::Handle
{
public:
	std::shared_ptr<TRN::Core::Batch> position;
};
class TRN::Core::Message::Payload<TRN::Core::Message::PERFORMANCES>::Handle
{
public:
	float cycles_per_second;
};

class TRN::Core::Message::Payload<TRN::Core::Message::TARGET_TRAJECTORY>::Handle
{
public:
	std::shared_ptr<TRN::Core::Matrix> trajectory;
};

class TRN::Core::Message::Payload<TRN::Core::Message::TARGET_SEQUENCE>::Handle
{
public:
	std::shared_ptr<TRN::Core::Matrix> sequence;
};


class TRN::Core::Message::Payload<TRN::Core::Message::STATES>::Handle
{
public:
	mutable std::size_t rows;
	mutable std::shared_ptr<TRN::Core::Matrix> global;
	mutable std::shared_ptr<TRN::Core::Matrix> reservoir;
	mutable std::shared_ptr<TRN::Core::Matrix> prediction;
	mutable std::shared_ptr<TRN::Core::Matrix> desired;
	mutable std::shared_ptr<TRN::Core::Matrix> stimulus;
};

class TRN::Core::Message::Payload<TRN::Core::Message::WEIGHTS>::Handle
{
public:
	mutable std::shared_ptr<TRN::Core::Matrix> feedfoward;
	mutable std::shared_ptr<TRN::Core::Matrix> recurrent;
	mutable std::shared_ptr<TRN::Core::Matrix> feedback;
	mutable std::shared_ptr<TRN::Core::Matrix> readout;
};

class TRN::Core::Message::Payload<TRN::Core::Message::CYCLES>::Handle
{
public:
	std::size_t batch_size;
	std::size_t cycles;
	std::size_t observations;
};

class TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>::Handle
{
public:
	size_t flops_per_epoch_factor;
	size_t flops_per_cycle;
};


class TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>::Handle
{
public:
	std::shared_ptr<TRN::Core::Scheduling> scheduling;
};

class TRN::Core::Message::Payload<TRN::Core::Message::SET>::Handle
{
public:
	std::string incoming;
	std::string expected;
	std::string label;
};

class TRN::Core::Message::Payload<TRN::Core::Message::TEST>::Handle
{
public:
	std::string label;
	std::size_t preamble;
};