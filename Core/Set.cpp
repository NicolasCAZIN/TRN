#include "stdafx.h"
#include "Set_impl.h"

TRN::Core::Set::Set(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<std::shared_ptr<TRN::Core::Matrix>> &sequences) :
	handle(std::make_unique<Handle>())
{
	switch (sequences.size())
	{
	case 0 :
		throw std::invalid_argument("Cannot create an empty batch");
	default :

		std::vector<float> concatenated;

		std::size_t concatenated_rows = 0;
		std::size_t concatenated_cols = sequences[0]->get_cols();
		std::vector<int> offsets;
		std::vector<int> durations;
		for (auto sequence : sequences)
		{
			std::vector<float> elements;
			std::size_t rows;
			std::size_t cols;
			sequence->to(elements, rows, cols);
			
			concatenated.insert(concatenated.end(), elements.begin(), elements.end());
		
			std::vector<int> sequence_offsets(rows);

			std::iota(sequence_offsets.begin(), sequence_offsets.end(), (int)concatenated_rows);
			
			offsets.insert(offsets.end(), sequence_offsets.begin(), sequence_offsets.end());
			concatenated_rows += rows;
			durations.push_back(sequence->get_rows());
		}
		handle->scheduling = TRN::Core::Scheduling::create(offsets, durations);
		handle->sequence = TRN::Core::Matrix::create(driver, concatenated, concatenated_rows, concatenated_cols);

		break;
	}

}

TRN::Core::Set::~Set()
{
	handle.reset();
}

const std::shared_ptr<TRN::Core::Matrix> &TRN::Core::Set::get_sequence()
{
	return handle->sequence;
}
const std::shared_ptr<TRN::Core::Scheduling> &TRN::Core::Set::get_scheduling()
{
	return handle->scheduling;
}

std::shared_ptr<TRN::Core::Set> TRN::Core::Set::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<std::shared_ptr<TRN::Core::Matrix>> &sequences)
{
	return std::make_shared<TRN::Core::Set>(driver, sequences);
}