#include "stdafx.h"
#include "Message_impl.h"

TRN::Core::Message::Payload<TRN::Core::Message::Type::STIMULUS>::Payload(const std::shared_ptr<TRN::Core::Batch> &stimulus, const unsigned long long &evaluation_id) :
	handle(std::make_unique<Handle>())
{
	handle->stimulus = stimulus;
	handle->evaluation_id = evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::STIMULUS>::Payload(const Payload<TRN::Core::Message::Type::STIMULUS> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->stimulus = ref.handle->stimulus;
	handle->evaluation_id = ref.handle->evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::STIMULUS>::~Payload()
{
	handle.reset();
}
const std::shared_ptr<TRN::Core::Batch> TRN::Core::Message::Payload<TRN::Core::Message::Type::STIMULUS>::get_stimulus() const
{
	return handle->stimulus;
}

const unsigned long long TRN::Core::Message::Payload<TRN::Core::Message::Type::STIMULUS>::get_evaluation_id() const
{
	return handle->evaluation_id;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::PREDICTION>::Payload(const std::shared_ptr<TRN::Core::Batch> &predicted, const unsigned long long &evaluation_id) :
	handle(std::make_unique<Handle>())
{
	handle->evaluation_id = evaluation_id;
	handle->predicted = predicted;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::PREDICTION>::Payload(const Payload<TRN::Core::Message::Type::PREDICTION> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->predicted = ref.handle->predicted;
	handle->evaluation_id = ref.handle->evaluation_id;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::PREDICTION>::~Payload()
{
	handle.reset();
}

const std::shared_ptr<TRN::Core::Batch> TRN::Core::Message::Payload<TRN::Core::Message::Type::PREDICTION>::get_predicted() const
{
	return handle->predicted;
}

const unsigned long long TRN::Core::Message::Payload<TRN::Core::Message::Type::PREDICTION>::get_evaluation_id() const
{
	return handle->evaluation_id;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::POSITION>::Payload(const std::shared_ptr<TRN::Core::Batch> &position,  const unsigned long long &evaluation_id) :
	handle(std::make_unique<Handle>())
{
	handle->evaluation_id = evaluation_id;
	handle->position = position;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::POSITION>::Payload(const Payload<TRN::Core::Message::Type::POSITION> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->position = ref.handle->position;
	handle->evaluation_id = ref.handle->evaluation_id;
}

const unsigned long long TRN::Core::Message::Payload<TRN::Core::Message::Type::POSITION>::get_evaluation_id() const
{
	return handle->evaluation_id;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::POSITION>::~Payload()
{
	handle.reset();
}

const std::shared_ptr<TRN::Core::Batch> TRN::Core::Message::Payload<TRN::Core::Message::Type::POSITION>::get_position() const
{
	return handle->position;
}



TRN::Core::Message::Payload<TRN::Core::Message::Type::PERFORMANCES>::Payload(const float &cycles_per_second) :
	handle(std::make_unique<Handle>())
{
	handle->cycles_per_second = cycles_per_second;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::PERFORMANCES>::Payload(const Payload<TRN::Core::Message::Type::PERFORMANCES> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->cycles_per_second = ref.handle->cycles_per_second;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::PERFORMANCES>::~Payload()
{
	handle.reset();
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::TARGET_TRAJECTORY>::Payload(const std::shared_ptr<TRN::Core::Matrix> &trajectory) :
	handle(std::make_unique<Handle>())
{
	handle->trajectory = trajectory;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TARGET_TRAJECTORY>::Payload(const Payload<TRN::Core::Message::Type::TARGET_TRAJECTORY> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->trajectory = ref.handle->trajectory;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TARGET_TRAJECTORY>::~Payload()
{
	handle.reset();
}

std::shared_ptr<TRN::Core::Matrix> TRN::Core::Message::Payload<TRN::Core::Message::Type::TARGET_TRAJECTORY>::get_trajectory() const
{
	return handle->trajectory;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::TARGET_SEQUENCE>::Payload(const std::shared_ptr<TRN::Core::Matrix> &sequence) :
	handle(std::make_unique<Handle>())
{
	handle->sequence = sequence;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TARGET_SEQUENCE>::Payload(const Payload<TRN::Core::Message::Type::TARGET_SEQUENCE> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->sequence = ref.handle->sequence;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TARGET_SEQUENCE>::~Payload()
{
	handle.reset();
}

std::shared_ptr<TRN::Core::Matrix> TRN::Core::Message::Payload<TRN::Core::Message::Type::TARGET_SEQUENCE>::get_sequence() const
{
	return handle->sequence;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::Payload() :
	handle(std::make_unique<Handle>())
{
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::Payload(const Payload<TRN::Core::Message::Type::STATES> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->rows = ref.handle->rows;
	handle->global = ref.handle->global;
	handle->reservoir = ref.handle->reservoir;
	handle->prediction = ref.handle->prediction;
	handle->desired = ref.handle->desired;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::~Payload()
{
	handle.reset();
}

 std::size_t &TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::get_rows() const
{
	return handle->rows;
}

 std::shared_ptr<TRN::Core::Matrix> TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::get_global() const
{
	return handle->global;
}
 std::shared_ptr<TRN::Core::Matrix> TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::get_reservoir() const
{
	return handle->reservoir;
}
 std::shared_ptr<TRN::Core::Matrix> TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::get_prediction() const
{
	return handle->prediction;
}
 std::shared_ptr<TRN::Core::Matrix> TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::get_desired() const
{
	return handle->desired;
}
 std::shared_ptr<TRN::Core::Matrix> TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::get_stimulus() const
 {
	 return handle->stimulus;
 }

void TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::set_rows(const std::size_t &rows) const
{
	handle->rows = rows;
}
void TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::set_global(const std::shared_ptr<TRN::Core::Matrix> &global) const
{
	handle->global = global;
}
void TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::set_reservoir(const std::shared_ptr<TRN::Core::Matrix> &reservoir) const
{
	handle->reservoir = reservoir;
}
void TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::set_prediction(const std::shared_ptr<TRN::Core::Matrix> &prediction) const
{
	handle->prediction = prediction;
}
void TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::set_desired(const std::shared_ptr<TRN::Core::Matrix> &desired) const
{
	handle->desired = desired;
}
void TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::set_stimulus(const std::shared_ptr<TRN::Core::Matrix> &stimulus) const
{
	handle->stimulus = stimulus;
}
std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>> TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>::create()
{
	return std::make_shared<TRN::Core::Message::Payload<TRN::Core::Message::Type::STATES>>();
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>::Payload() :
	handle(std::make_unique<Handle>())
{
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>::Payload(const Payload<TRN::Core::Message::Type::WEIGHTS> &ref) :
handle(std::make_unique<Handle>())
{
	handle->feedfoward = ref.handle->feedfoward;
	handle->feedback = ref.handle->feedback;
	handle->recurrent = ref.handle->recurrent;
	handle->readout = ref.handle->readout;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>::~Payload()
{
	handle.reset();
}
const std::shared_ptr<TRN::Core::Batch> TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>::get_feedforward() const
{
	return handle->feedfoward;
}
const std::shared_ptr<TRN::Core::Batch> TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>::get_recurrent() const
{
	return handle->recurrent;
}
const std::shared_ptr<TRN::Core::Batch> TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>::get_readout() const
{
	return handle->readout;
}
void TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>::set_feedforward(const std::shared_ptr<TRN::Core::Batch> &feedforward) const
{
	handle->feedfoward = feedforward;
}
void TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>::set_recurrent(const std::shared_ptr<TRN::Core::Batch> &recurrent) const
{
	handle->recurrent = recurrent;
}
void TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>::set_readout(const std::shared_ptr<TRN::Core::Batch> &readout) const
{
	handle->readout = readout;
}

std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>> TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>::create()
{
	return std::make_shared<TRN::Core::Message::Payload<TRN::Core::Message::Type::WEIGHTS>>();
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::TESTED>::Payload(const unsigned long long &evaluation_id) :
	handle(std::make_unique<Handle>())
{
	handle->evaluation_id = evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TESTED>::Payload(const Payload<TRN::Core::Message::Type::TESTED> &ref):
	handle(std::make_unique<Handle>())
{
	handle->evaluation_id = ref.handle->evaluation_id;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::TESTED>::~Payload()
{
	handle.reset();
}

const unsigned long long &TRN::Core::Message::Payload<TRN::Core::Message::Type::TESTED>::get_evaluation_id() const
{
	return handle->evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::PRIMED>::Payload(const unsigned long long &evaluation_id) :
	handle(std::make_unique<Handle>())
{
	handle->evaluation_id = evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::PRIMED>::Payload(const Payload<TRN::Core::Message::Type::PRIMED> &ref):
handle(std::make_unique<Handle>())
{
	handle->evaluation_id = ref.handle->evaluation_id;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::PRIMED>::~Payload()
{
	handle.reset();
}
const unsigned long long &TRN::Core::Message::Payload<TRN::Core::Message::Type::PRIMED>::get_evaluation_id() const
{
	return handle->evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TRAINED>::Payload(const unsigned long long &evaluation_id) :
	handle(std::make_unique<Handle>())
{
	handle->evaluation_id = evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TRAINED>::Payload(const Payload<TRN::Core::Message::Type::TRAINED> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->evaluation_id = ref.handle->evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TRAINED>::~Payload()
{
	handle.reset();
}
const unsigned long long &TRN::Core::Message::Payload<TRN::Core::Message::Type::TRAINED>::get_evaluation_id() const
{
	return handle->evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::CONFIGURED>::Payload()
{
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::CONFIGURED>::Payload(const Payload<TRN::Core::Message::Type::CONFIGURED> &ref)
{

}
TRN::Core::Message::Payload<TRN::Core::Message::Type::CONFIGURED>::~Payload()
{
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::CYCLES>::Payload(const std::size_t &batch_size, const std::size_t &cycles, const std::size_t &observations) :
	handle(std::make_unique<Handle>())
{
	handle->batch_size = batch_size;
	handle->cycles = cycles;
	handle->observations = observations;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::CYCLES>::~Payload()
{
	handle.reset();
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::CYCLES>::Payload(const Payload<TRN::Core::Message::Type::CYCLES> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->batch_size = ref.handle->batch_size;
	handle->cycles = ref.handle->cycles;
	handle->observations = ref.handle->observations;
}
const std::size_t &TRN::Core::Message::Payload<TRN::Core::Message::Type::CYCLES>::get_batch_size() const
{
	return handle->batch_size;
}
const std::size_t &TRN::Core::Message::Payload<TRN::Core::Message::Type::CYCLES>::get_cycles() const
{
	return handle->cycles;
}
const std::size_t &TRN::Core::Message::Payload<TRN::Core::Message::Type::CYCLES>::get_observations() const
{
	return handle->observations;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::FLOPS>::Payload(const size_t &flops_per_epoch_factor, const size_t &flops_per_cycle) :
	handle(std::make_unique<Handle>())
{
	handle->flops_per_epoch_factor = flops_per_epoch_factor;
	handle->flops_per_cycle = flops_per_cycle;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::FLOPS>::~Payload()
{
	handle.reset();
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::FLOPS>::Payload(const Payload<TRN::Core::Message::Type::FLOPS> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->flops_per_epoch_factor = ref.handle->flops_per_epoch_factor;
	handle->flops_per_cycle = ref.handle->flops_per_cycle;
}

const std::size_t &TRN::Core::Message::Payload<TRN::Core::Message::Type::FLOPS>::get_flops_per_epoch_factor() const
{
	return handle->flops_per_epoch_factor;
}

const std::size_t &TRN::Core::Message::Payload<TRN::Core::Message::Type::FLOPS>::get_flops_per_cycle() const
{
	return handle->flops_per_cycle;
}

void TRN::Core::Message::Payload<TRN::Core::Message::Type::FLOPS>::set_flops_per_epoch_factor(const std::size_t &flops_per_epoch_factor) const
{
	handle->flops_per_epoch_factor = flops_per_epoch_factor;
}

void TRN::Core::Message::Payload<TRN::Core::Message::Type::FLOPS>::set_flops_per_cycle(const std::size_t &flops_per_cycle) const
{
	handle->flops_per_cycle = flops_per_cycle;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::SCHEDULING>::Payload(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Scheduling> &scheduling) :
	handle(std::make_unique<Handle>())
{
	handle->scheduling = scheduling;
	handle->evaluation_id = evaluation_id;

}
TRN::Core::Message::Payload<TRN::Core::Message::Type::SCHEDULING>::Payload(const Payload<TRN::Core::Message::Type::SCHEDULING> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->scheduling = ref.handle->scheduling;
	handle->evaluation_id = ref.handle->evaluation_id;

}

TRN::Core::Message::Payload<TRN::Core::Message::Type::SCHEDULING>::~Payload()
{
	handle.reset();
}
const std::shared_ptr<TRN::Core::Scheduling> TRN::Core::Message::Payload<TRN::Core::Message::Type::SCHEDULING>::get_scheduling() const
{
	return handle->scheduling;
}

const unsigned long long TRN::Core::Message::Payload<TRN::Core::Message::Type::SCHEDULING>::get_evaluation_id() const
{
	return handle->evaluation_id;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::SET>::Payload(const std::string &label, const std::string &incoming, const std::string &expected, const unsigned long long  &evaluation_id) :
	handle(std::make_unique<Handle>())
{
	handle->label = label;
	handle->incoming = incoming;
	handle->expected = expected;
	handle->evaluation_id = evaluation_id;
}

TRN::Core::Message::Payload<TRN::Core::Message::Type::SET>::Payload(const Payload<TRN::Core::Message::Type::SET> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->label = ref.handle->label;
	handle->incoming = ref.handle->incoming;
	handle->expected = ref.handle->expected;
	handle->evaluation_id = ref.handle->evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::SET>::~Payload()
{
	handle.reset();
}

const std::string &TRN::Core::Message::Payload<TRN::Core::Message::Type::SET>::get_label() const
{
	return handle->label;
}
const std::string &TRN::Core::Message::Payload<TRN::Core::Message::Type::SET>::get_incoming() const
{
	return handle->incoming;
}
const std::string &TRN::Core::Message::Payload<TRN::Core::Message::Type::SET>::get_expected() const
{
	return handle->expected;
}
const unsigned long long &TRN::Core::Message::Payload<TRN::Core::Message::Type::SET>::get_evaluation_id() const
{
	return handle->evaluation_id;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TEST>::Payload(const std::string &label, const bool &autonomous, const std::size_t &preamble, const std::size_t &supplementary_generations) :
	handle(std::make_unique<Handle>())
{
	handle->label = label;
	handle->preamble = preamble;
	handle->autonomous = autonomous;
	handle->supplementary_generations = supplementary_generations;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TEST>::Payload(const Payload<TRN::Core::Message::Type::TEST> &ref) :
	handle(std::make_unique<Handle>())
{
	handle->label = ref.handle->label;
	handle->preamble = ref.handle->preamble;
	handle->autonomous = ref.handle->autonomous;
	handle->supplementary_generations = ref.handle->supplementary_generations;
}
TRN::Core::Message::Payload<TRN::Core::Message::Type::TEST>::~Payload()
{
	handle.reset();
}

const std::string &TRN::Core::Message::Payload<TRN::Core::Message::Type::TEST>::get_label() const
{
	return handle->label;
}
const std::size_t &TRN::Core::Message::Payload<TRN::Core::Message::Type::TEST>::get_preamble() const
{
	return handle->preamble;
}
const bool &TRN::Core::Message::Payload<TRN::Core::Message::Type::TEST>::get_autonomous() const
{
	return handle->autonomous;
}
const std::size_t &TRN::Core::Message::Payload<TRN::Core::Message::Type::TEST>::get_supplementary_generations() const
{
	return handle->supplementary_generations;
}
