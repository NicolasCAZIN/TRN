#pragma once
#include "Sequence.h"

class TRN::Core::Sequence::Handle
{
public:
	std::shared_ptr<TRN::Core::Matrix> incoming;
	std::shared_ptr<TRN::Core::Matrix> expected;
	std::shared_ptr<TRN::Core::Matrix> reward;
	std::string name;
};
