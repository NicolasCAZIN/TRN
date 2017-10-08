#include "stdafx.h"
#include "Driver.h"
#include "CPU/Driver.h"
#include "GPU/Driver.h"



std::shared_ptr<TRN::Backend::Driver> TRN::Model::Driver::create(const int &index)
{
	// std::cout << __FUNCTION__ << std::endl;
	switch (index)
	{
	case 0:
	{
		std::string brand;
		TRN::CPU::Implementation implementation;
		TRN::CPU::query(brand, implementation);

		std::cout << "CPU version selected : " << brand << std::endl;
		switch (implementation)
		{
#if !defined(_M_IX86) && (defined(_M_AMD64) ||defined(_M_X64))
			case TRN::CPU::FMA3 :
				return TRN::CPU::Driver<TRN::CPU::FMA3>::create();
			case TRN::CPU::AVX2:
				return TRN::CPU::Driver<TRN::CPU::AVX2>::create();
			case TRN::CPU::AVX:
				return TRN::CPU::Driver<TRN::CPU::AVX>::create();
			case TRN::CPU::SSE41:
				return TRN::CPU::Driver<TRN::CPU::SSE41>::create();
			case TRN::CPU::SSE3:
				return TRN::CPU::Driver<TRN::CPU::SSE3>::create();
			case TRN::CPU::SSE2:
				return TRN::CPU::Driver<TRN::CPU::SSE2>::create();
#endif 
#if (defined(_M_IX86) && !defined(_M_AMD64)  && !defined(_M_X64))
			case TRN::CPU::MMX_SSE:
				return TRN::CPU::Driver<TRN::CPU::SSE2>::create();
#endif
			case TRN::CPU::SCALAR:
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