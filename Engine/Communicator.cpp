#include "stdafx.h"
#include "Communicator_impl.h"
#include "Helper/Logger.h"

TRN::Engine::Communicator::Communicator() :
	handle(std::make_unique<Handle>())
{
	TRACE_LOGGER;
	handle->host = boost::asio::ip::host_name();
}

TRN::Engine::Communicator::~Communicator()
{
	TRACE_LOGGER;
	handle.reset();
}

std::string TRN::Engine::Communicator::host()
{
	TRACE_LOGGER;
	return handle->host;
}



template <TRN::Engine::Tag tag>
void TRN::Engine::Communicator::send(const TRN::Engine::Message<tag> &message, const int &destination)
{
	TRACE_LOGGER;
	std::ostringstream archive_stream;
	boost::archive::binary_oarchive archive(archive_stream);
	archive << message;

	auto to_send = archive_stream.str();

	send(destination, tag, compress<RAW>(to_send));
}
template <TRN::Engine::Tag tag>
void TRN::Engine::Communicator::broadcast(const TRN::Engine::Message<tag> &message)
{
	TRACE_LOGGER;
	for (int rank = 1; rank < size(); rank++)
	{
		send(message, rank);
	}
}
template <TRN::Engine::Tag tag>
TRN::Engine::Message<tag> TRN::Engine::Communicator::receive(const int &destination)
{
	TRACE_LOGGER;
	auto received = receive(destination, tag);
	std::string data = decompress<RAW>(received);

	std::istringstream archive_stream(data);
	boost::archive::binary_iarchive archive(archive_stream);

	TRN::Engine::Message<tag> message;

	archive >> message;

	return message;
}
template void TRN::Engine::Communicator::broadcast(const TRN::Engine::Message<TRN::Engine::Tag::STOP> &message);
template void TRN::Engine::Communicator::broadcast(const TRN::Engine::Message<TRN::Engine::Tag::START> &message);
template void TRN::Engine::Communicator::broadcast(const TRN::Engine::Message<TRN::Engine::Tag::QUIT> &message);
template TRN::Engine::Message<TRN::Engine::Tag::TERMINATED> TRN::Engine::Communicator::receive<TRN::Engine::Tag::TERMINATED>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::EXIT> TRN::Engine::Communicator::receive<TRN::Engine::Tag::EXIT>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::QUIT> TRN::Engine::Communicator::receive<TRN::Engine::Tag::QUIT>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::START> TRN::Engine::Communicator::receive<TRN::Engine::Tag::START>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::STOP> TRN::Engine::Communicator::receive<TRN::Engine::Tag::STOP>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::ALLOCATED> TRN::Engine::Communicator::receive<TRN::Engine::Tag::ALLOCATED>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATED> TRN::Engine::Communicator::receive<TRN::Engine::Tag::DEALLOCATED>(const int &destination);
//template TRN::Engine::Message<TRN::Engine::Tag::READY> TRN::Engine::Communicator::receive<TRN::Engine::Tag::READY>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURED> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURED>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::WORKER> TRN::Engine::Communicator::receive<TRN::Engine::Tag::WORKER>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CACHED> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CACHED>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::ALLOCATE> TRN::Engine::Communicator::receive<TRN::Engine::Tag::ALLOCATE>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATE> TRN::Engine::Communicator::receive<TRN::Engine::Tag::DEALLOCATE>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::TRAIN> TRN::Engine::Communicator::receive<TRN::Engine::Tag::TRAIN>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::TEST> TRN::Engine::Communicator::receive<TRN::Engine::Tag::TEST>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SEQUENCE> TRN::Engine::Communicator::receive<TRN::Engine::Tag::DECLARE_SEQUENCE>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SET> TRN::Engine::Communicator::receive<TRN::Engine::Tag::DECLARE_SET>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::SETUP_STATES> TRN::Engine::Communicator::receive<TRN::Engine::Tag::SETUP_STATES>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::SETUP_WEIGHTS> TRN::Engine::Communicator::receive<TRN::Engine::Tag::SETUP_WEIGHTS>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::SETUP_PERFORMANCES> TRN::Engine::Communicator::receive<TRN::Engine::Tag::SETUP_PERFORMANCES>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::SETUP_SCHEDULING> TRN::Engine::Communicator::receive<TRN::Engine::Tag::SETUP_SCHEDULING>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_BEGIN> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_BEGIN>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_END> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_END>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_ENCODER_MODEL> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_ENCODER_MODEL>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_ENCODER_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_ENCODER_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_DECODER_LINEAR> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_DECODER_LINEAR>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MAP> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MAP>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MODEL> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MODEL>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_LOOP_COPY>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::POSITION> TRN::Engine::Communicator::receive<TRN::Engine::Tag::POSITION>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::STIMULUS> TRN::Engine::Communicator::receive<TRN::Engine::Tag::STIMULUS>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::SCHEDULING> TRN::Engine::Communicator::receive<TRN::Engine::Tag::SCHEDULING>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> TRN::Engine::Communicator::receive<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> TRN::Engine::Communicator::receive<TRN::Engine::Tag::RECURRENT_WEIGHTS>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::READOUT_WEIGHTS> TRN::Engine::Communicator::receive<TRN::Engine::Tag::READOUT_WEIGHTS>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::MUTATOR_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::MUTATOR_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::SCHEDULER_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::SCHEDULER_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_DIMENSIONS> TRN::Engine::Communicator::receive<TRN::Engine::Tag::FEEDFORWARD_DIMENSIONS>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_DIMENSIONS> TRN::Engine::Communicator::receive<TRN::Engine::Tag::RECURRENT_DIMENSIONS>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::READOUT_DIMENSIONS> TRN::Engine::Communicator::receive<TRN::Engine::Tag::READOUT_DIMENSIONS>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::STATES> TRN::Engine::Communicator::receive<TRN::Engine::Tag::STATES>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::WEIGHTS> TRN::Engine::Communicator::receive<TRN::Engine::Tag::WEIGHTS>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::PERFORMANCES> TRN::Engine::Communicator::receive<TRN::Engine::Tag::PERFORMANCES>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> TRN::Engine::Communicator::receive<TRN::Engine::Tag::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_READOUT_FRECHET_DISTANCE> TRN::Engine::Communicator::receive<TRN::Engine::Tag::MEASUREMENT_READOUT_FRECHET_DISTANCE>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_READOUT_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::MEASUREMENT_READOUT_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> TRN::Engine::Communicator::receive<TRN::Engine::Tag::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_POSITION_FRECHET_DISTANCE> TRN::Engine::Communicator::receive<TRN::Engine::Tag::MEASUREMENT_POSITION_FRECHET_DISTANCE>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_POSITION_CUSTOM> TRN::Engine::Communicator::receive<TRN::Engine::Tag::MEASUREMENT_POSITION_CUSTOM>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::LOG_INFORMATION> TRN::Engine::Communicator::receive<TRN::Engine::Tag::LOG_INFORMATION>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::LOG_WARNING> TRN::Engine::Communicator::receive<TRN::Engine::Tag::LOG_WARNING>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::LOG_ERROR> TRN::Engine::Communicator::receive<TRN::Engine::Tag::LOG_ERROR>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::TRAINED> TRN::Engine::Communicator::receive<TRN::Engine::Tag::TRAINED>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::PRIMED> TRN::Engine::Communicator::receive<TRN::Engine::Tag::PRIMED>(const int &destination);
template TRN::Engine::Message<TRN::Engine::Tag::TESTED> TRN::Engine::Communicator::receive<TRN::Engine::Tag::TESTED>(const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURED> &message, const int &destination);
//template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::READY> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::START> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::STOP> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::QUIT> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::EXIT> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::TERMINATED> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::ALLOCATED> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATED> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::WORKER> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CACHED> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::ALLOCATE> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATE> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::TRAIN> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::TEST> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SEQUENCE> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SET> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_STATES> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_WEIGHTS> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_PERFORMANCES> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_SCHEDULING> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_BEGIN> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_END> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_ENCODER_MODEL> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_ENCODER_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_DECODER_LINEAR> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MAP> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MODEL> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::POSITION> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::STIMULUS> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::SCHEDULING> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::READOUT_WEIGHTS> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::MUTATOR_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::SCHEDULER_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_DIMENSIONS> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_DIMENSIONS> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::READOUT_DIMENSIONS> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::STATES> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::WEIGHTS> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::PERFORMANCES> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::TRAINED> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::PRIMED> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::TESTED> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_READOUT_FRECHET_DISTANCE> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_READOUT_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_POSITION_FRECHET_DISTANCE> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::MEASUREMENT_POSITION_CUSTOM> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::LOG_INFORMATION> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::LOG_WARNING> &message, const int &destination);
template void TRN::Engine::Communicator::send(const TRN::Engine::Message<TRN::Engine::Tag::LOG_ERROR> &message, const int &destination);