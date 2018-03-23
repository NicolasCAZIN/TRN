#include "stdafx.h"
#include "Messages.h"

std::ostream &operator << (std::ostream &os, const TRN::Engine::Tag &tag)
{
	switch (tag)
	{
		case TRN::Engine::Tag::QUIT:
			os << "QUIT";
			break;
		case TRN::Engine::Tag::EXIT:
			os << "EXIT";
			break;
		case TRN::Engine::Tag::TERMINATED:
			os << "TERMINATED";
			break;
		case TRN::Engine::Tag::STOP:
			os << "STOP";
			break;
		case TRN::Engine::Tag::START:
			os << "START";
			break;
		case TRN::Engine::Tag::WORKER:
			os << "WORKER";
			break;
		case TRN::Engine::Tag::ALLOCATE:
			os << "ALLOCATE";
			break;
		case TRN::Engine::Tag::DEALLOCATE:
			os << "DEALLOCATE";
			break;
		case TRN::Engine::Tag::TRAIN:
			os << "TRAIN";
			break;
		case TRN::Engine::Tag::TEST:
			os << "TEST";
			break;
		case TRN::Engine::Tag::DECLARE_SEQUENCE:
			os << "DECLARE_SEQUENCE";
			break;
		case TRN::Engine::Tag::DECLARE_SET:
			os << "DECLARE_SET";
			break;
		case TRN::Engine::Tag::SETUP_STATES:
			os << "SETUP_STATES";
			break;
		case TRN::Engine::Tag::SETUP_WEIGHTS:
			os << "SETUP_WEIGHTS";
			break;
		case TRN::Engine::Tag::SETUP_PERFORMANCES:
			os << "SETUP_PERFORMANCES";
			break;
		case TRN::Engine::Tag::SETUP_SCHEDULING:
			os << "SETUP_SCHEDULING";
			break;
		case TRN::Engine::Tag::CONFIGURE_BEGIN:
			os << "CONFIGURE_BEGIN";
			break;
		case TRN::Engine::Tag::CONFIGURE_END:
			os << "CONFIGURE_END";
			break;
		case TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR:
			os << "CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR";
			break;
		case TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE:
			os << "CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE";
			break;
		case TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM:
			os << "CONFIGURE_MEASUREMENT_READOUT_CUSTOM";
			break;
		case TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR:
			os << "CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR";
			break;
		case TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE:
			os << "CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE";
			break;
		case TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM:
			os << "CONFIGURE_MEASUREMENT_POSITION_CUSTOM";
			break;
		case TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF:
			os << "CONFIGURE_RESERVOIR_WIDROW_HOFF";
			break;
		case TRN::Engine::Tag::CONFIGURE_DECODER_LINEAR:
			os << "CONFIGURE_DECODER_LINEAR";
			break;
		case TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MODEL:
			os << "CONFIGURE_DECODER_KERNEL_MODEL";
			break;
		case TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MAP:
			os << "CONFIGURE_DECODER_KERNEL_MAP";
			break;

		case TRN::Engine::Tag::CONFIGURE_LOOP_COPY:
			os << "CONFIGURE_LOOP_COPY";
			break;
		case TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER:
			os << "CONFIGURE_LOOP_SPATIAL_FILTER";
			break;
		case TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM:
			os << "CONFIGURE_LOOP_CUSTOM";
			break;
		case TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED:
			os << "CONFIGURE_SCHEDULER_TILED";
			break;
		case TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS:
			os << "CONFIGURE_SCHEDULER_SNIPPETS";
			break;
		case TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM:
			os << "CONFIGURE_SCHEDULER_CUSTOM";
			break;
		case TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE:
			os << "CONFIGURE_MUTATOR_SHUFFLE";
			break;
		case TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE:
			os << "CONFIGURE_MUTATOR_REVERSE";
			break;
		case TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH:
			os << "CONFIGURE_MUTATOR_PUNCH";
			break;
		case TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM:
			os << "CONFIGURE_MUTATOR_CUSTOM";
			break;
		case TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM:
			os << "CONFIGURE_FEEDFORWARD_UNIFORM";
			break;
		case TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN:
			os << "CONFIGURE_FEEDFORWARD_GAUSSIAN";
			break;
		case TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM:
			os << "CONFIGURE_FEEDFORWARD_CUSTOM";
			break;
		case TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM:
			os << "CONFIGURE_RECURRENT_UNIFORM";
			break;
		case TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN:
			os << "CONFIGURE_RECURRENT_GAUSSIAN";
			break;
		case TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM:
			os << "CONFIGURE_RECURRENT_CUSTOM";
			break;
	
		case TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM:
			os << "CONFIGURE_READOUT_UNIFORM";
			break;
		case TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN:
			os << "CONFIGURE_READOUT_GAUSSIAN";
			break;
		case TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM:
			os << "CONFIGURE_READOUT_CUSTOM";
			break;
		case TRN::Engine::Tag::POSITION:
			os << "POSITION";
			break;
		case TRN::Engine::Tag::STIMULUS:
			os << "STIMULUS";
			break;
		case TRN::Engine::Tag::SCHEDULING:
			os << "SCHEDULING";
			break;
		case TRN::Engine::Tag::FEEDFORWARD_WEIGHTS:
			os << "FEEDFORWARD_WEIGHTS";
			break;
		case TRN::Engine::Tag::RECURRENT_WEIGHTS:
			os << "RECURRENT_WEIGHTS";
			break;
		
		case TRN::Engine::Tag::READOUT_WEIGHTS:
			os << "READOUT_WEIGHTS";
			break;
		case TRN::Engine::Tag::MUTATOR_CUSTOM:
			os << "MUTATOR_CUSTOM";
			break;
		case TRN::Engine::Tag::SCHEDULER_CUSTOM:
			os << "SCHEDULER_CUSTOM";
			break;
		case TRN::Engine::Tag::FEEDFORWARD_DIMENSIONS:
			os << "FEEDFORWARD_DIMENSIONS";
			break;
		case TRN::Engine::Tag::RECURRENT_DIMENSIONS:
			os << "RECURRENT_DIMENSIONS";
			break;
	
		case TRN::Engine::Tag::READOUT_DIMENSIONS:
			os << "READOUT_DIMENSIONS";
			break;
		case TRN::Engine::Tag::STATES:
			os << "STATES";
			break;
		case TRN::Engine::Tag::WEIGHTS:
			os << "WEIGHTS";
			break;
		case TRN::Engine::Tag::PERFORMANCES:
			os << "PERFORMANCES";
			break;
		case TRN::Engine::Tag::CONFIGURED:
			os << "CONFIGURED";
			break;
		case TRN::Engine::Tag::TRAINED:
			os << "TRAINED";
			break;
		case TRN::Engine::Tag::TESTED:
			os << "TESTED";
			break;
		case TRN::Engine::Tag::PRIMED:
			os << "PRIMED";
			break;
		case TRN::Engine::Tag::ALLOCATED:
			os << "ALLOCATED";
			break;
		case TRN::Engine::Tag::DEALLOCATED:
			os << "DEALLOCATED";
			break;
		case TRN::Engine::Tag::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR:
			os << "MEASUREMENT_READOUT_MEAN_SQUARE_ERROR";
			break;
		case TRN::Engine::Tag::MEASUREMENT_READOUT_FRECHET_DISTANCE:
			os << "MEASUREMENT_READOUT_FRECHET_DISTANCE";
			break;
		case TRN::Engine::Tag::MEASUREMENT_READOUT_CUSTOM:
			os << "MEASUREMENT_READOUT_CUSTOM";
			break;
		case TRN::Engine::Tag::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR:
			os << "MEASUREMENT_POSITION_MEAN_SQUARE_ERROR";
			break;
		case TRN::Engine::Tag::MEASUREMENT_POSITION_FRECHET_DISTANCE:
			os << "MEASUREMENT_POSITION_FRECHET_DISTANCE";
			break;
		case TRN::Engine::Tag::MEASUREMENT_POSITION_CUSTOM:
			os << "MEASUREMENT_POSITION_CUSTOM";
			break;
		case TRN::Engine::Tag::LOG_INFORMATION:
			os << "LOG_INFORMATION";
			break;
		case TRN::Engine::Tag::LOG_WARNING:
			os << "LOG_WARNING";
			break;
		case TRN::Engine::Tag::LOG_ERROR:
			os << "LOG_ERROR";
			break;
	}

	return os;
}

unsigned int TRN::Engine::checksum(const std::vector<float> &sequence)
{
	const void *buffer = sequence.data();
	std::size_t length = sequence.size() * sizeof(float);

	boost::crc_32_type crc;
	crc.process_bytes(buffer, length);
	return crc.checksum();
}

namespace TRN::Engine
{
	union Identifier
	{
		struct
		{
			unsigned long long batch_number : 32;
			unsigned long long condition_number : 16;
			unsigned long long frontend_number : 16;
		};

		unsigned long long id;
	};
};

void   TRN::Engine::encode(const unsigned short &frontend_number, const unsigned short &condition_number, const unsigned int &batch_number, unsigned long long &simulation_id)
{
	Identifier identifier;

	identifier.frontend_number = frontend_number;
	identifier.condition_number = condition_number;
	identifier.batch_number = batch_number;
	simulation_id = identifier.id;
}

void TRN::Engine::decode(const unsigned long long &simulation_id, unsigned short &frontend_number, unsigned short &condition_number, unsigned int &batch_number)
{
	Identifier identifier;

	identifier.id = simulation_id;
	frontend_number = identifier.frontend_number;
	condition_number = identifier.condition_number;
	batch_number = identifier.batch_number;
}

namespace TRN::Engine::Evaluation
{
	union Identifier
	{
		struct
		{
			unsigned long long trial_number : 16;
			unsigned long long train_number : 16;
			unsigned long long test_number : 16;
			unsigned long long repeat_number : 16;
		};

		unsigned long long id;
	};
};

void   TRN::Engine::Evaluation::encode(const unsigned short &trial_number, const unsigned short &train_number, const unsigned short &test_number, const unsigned short &repeat_number, unsigned long long &evaluation_id)
{
	Identifier identifier;

	identifier.trial_number = trial_number;
	identifier.train_number = train_number;
	identifier.test_number = test_number;
	identifier.repeat_number = repeat_number;
	evaluation_id = identifier.id;
}

void TRN::Engine::Evaluation::decode(const unsigned long long &simulation_id, unsigned short &trial_number, unsigned short &train_number, unsigned short &test_number, unsigned short &repeat_number)
{
	Identifier identifier;

	identifier.id = simulation_id;
	trial_number = identifier.trial_number;
	train_number = identifier.train_number;
	test_number = identifier.test_number;
	repeat_number = identifier.repeat_number;
}