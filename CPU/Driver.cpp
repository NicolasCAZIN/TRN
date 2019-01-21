#include "stdafx.h"
#include "Driver.h"
#include "Memory.h"
#include "Random.h"
#include "Algorithm.h"

template<TRN::CPU::Implementation Implementation>
TRN::CPU::Driver<Implementation>::Driver() :
	TRN::Backend::Driver(TRN::CPU::Memory<Implementation>::create(), TRN::CPU::Random::create(), TRN::CPU::Algorithm<Implementation>::create())
{

}
template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Driver<Implementation>::dispose()
{

}

template<TRN::CPU::Implementation Implementation>
std::shared_ptr<TRN::CPU::Driver<Implementation>> TRN::CPU::Driver<Implementation>::create()
{
	return std::make_shared<TRN::CPU::Driver<Implementation>>();
}



template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Driver<Implementation>::synchronize()
{

}
template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Driver<Implementation>::toggle()
{

}
template TRN::CPU::Driver<TRN::CPU::Implementation::SCALAR>;
#if (!defined(_M_IX86) && (defined(_M_AMD64) || defined(_M_X64)))
template TRN::CPU::Driver<TRN::CPU::Implementation::SSE2>;
template TRN::CPU::Driver<TRN::CPU::Implementation::SSE3>;
template TRN::CPU::Driver<TRN::CPU::Implementation::SSE41>;
template TRN::CPU::Driver<TRN::CPU::Implementation::AVX>;
template TRN::CPU::Driver<TRN::CPU::Implementation::AVX2_FMA3>;

#endif

#if (defined(_M_IX86) && !defined(_M_AMD64) && !defined(_M_X64))
template TRN::CPU::Driver<TRN::CPU::Implementation::MMX_SSE>;
#endif
template<TRN::CPU::Implementation Implementation>
std::string  TRN::CPU::Driver<Implementation>::name()
{
	std::string brand;
	TRN::CPU::Implementation implementation;
	TRN::CPU::query(brand, implementation);


	return brand;
}
template<TRN::CPU::Implementation Implementation>
int TRN::CPU::Driver<Implementation>::index()
{
	return 0;
}

static std::list<std::pair<int, std::string>> enumerated;

std::list<std::pair<int, std::string>> TRN::CPU::enumerate_devices()
{
	if (enumerated.empty())
	{

		std::string brand;
		TRN::CPU::Implementation implementation;
		TRN::CPU::query(brand, implementation);

		enumerated.push_back(std::make_pair(0, brand));
	}
	return enumerated;
}