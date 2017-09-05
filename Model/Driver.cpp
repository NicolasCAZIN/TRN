#include "stdafx.h"
#include "Driver.h"
#include "CPU/InstructionSet.h"
#include "CPU/Driver.h"
#include "GPU/Driver.h"



std::shared_ptr<TRN::Backend::Driver> TRN::Model::Driver::create(const int &index)
{
	switch (index)
	{
	case 0:
	{
		std::cout << "CPU version selected : " << TRN::CPU::InstructionSet::Brand() << std::endl;
#if !defined(_M_IX86) && (defined(_M_AMD64) ||defined(_M_X64))
		if (TRN::CPU::InstructionSet::FMA()) //hadd 256_dp_ps
		{
			std::cout << "FMA3 implementation selected" << std::endl;
			return TRN::CPU::Driver<TRN::CPU::FMA3>::create();
		}
		else if (TRN::CPU::InstructionSet::AVX2()) //hadd 256_dp_ps
		{
			std::cout << "AVX2 implementation selected" << std::endl;
			return TRN::CPU::Driver<TRN::CPU::AVX2>::create();
		}
		else if (TRN::CPU::InstructionSet::AVX()) //hadd 256_dp_ps
		{
				std::cout << "AVX implementation selected" << std::endl;
				return TRN::CPU::Driver<TRN::CPU::AVX>::create();
		}
		else if (TRN::CPU::InstructionSet::SSE41()) // hadd
		{
			std::cout << "SSE41 implementation selected" << std::endl;
			return TRN::CPU::Driver<TRN::CPU::SSE41>::create();
		}
		else if (TRN::CPU::InstructionSet::SSE3()) // hadd
		{
			std::cout << "SSE3 implementation selected" << std::endl;
			return TRN::CPU::Driver<TRN::CPU::SSE3>::create();
		}
		else if (TRN::CPU::InstructionSet::SSE2())
		{
			std::cout << "SSE2 implementation selected" << std::endl;
			return TRN::CPU::Driver<TRN::CPU::SSE2>::create();
		}
#endif 
#if (defined(_M_IX86) && !defined(_M_AMD64)  && !defined(_M_X64))
		else  if (InstructionSet::MMX() && InstructionSet::SSE())
		{
			std::cout << "MMX+SSE implementation selected" << std::endl;
			return TRN::CPU::Driver<TRN::CPU::MMX_SSE>::create();
		}
#endif
		else
		{
				//throw std::runtime_error("SCALAR fallback not yet fixed");
			std::cout << "SCALAR implementation selected" << std::endl;
			return TRN::CPU::Driver<TRN::CPU::SCALAR>::create();
		}
	}
		
	default :
		return TRN::GPU::Driver::create(index - 1);
	}
}




 std::list<std::pair<int, std::string>> TRN::Model::Driver::enumerate_devices()
{
	std::list<std::pair<int, std::string>> enumerated = TRN::CPU::enumerate_devices();

	for (auto device : TRN::GPU::enumerate_devices())
	{
		enumerated.push_back(device);
	}
	return enumerated;
}