#include "stdafx.h"
#include "Implementation.h"

std::ostream &operator << (std::ostream &stream, const TRN::CPU::Implementation &implementation)
{
	
	switch (implementation)
	{
		case	TRN::CPU::Implementation::SCALAR :
			stream << "SCALAR";
			break;
#if defined(_M_IX86) && !defined(_M_X64)
		case	TRN::CPU::Implementation::MMX_SSE : //exp_sp compiled with MMX
			stream << "MMX_SSE";
			break;
#endif
#if !defined(_M_IX86) && defined(_M_X64)
		case	TRN::CPU::Implementation::SSE2 : //exp_sp compiled with SSE2
			stream << "SSE2";
			break;
		case	TRN::CPU::Implementation::SSE3 : //hadd, exp_sp compiled with SSE2
			stream << "SSE3";
			break;
		case	TRN::CPU::Implementation::SSE41 :// dp_ps, exp_ps compiled with SSE2
			stream << "SSE41";
			break;
		case	TRN::CPU::Implementation::AVX : // 256_dp_ps, exp256_ps compiled with SSE2
			stream << "AVX";
			break;
		case	TRN::CPU::Implementation::AVX2_FMA3 : // exp256_ps compiled with avx2
			stream << "AVX2+FMA3";
			break;
		default :
			throw std::runtime_error("Unexpected implementation type " + std::to_string(implementation));
#endif
	}
	return stream;
}

void TRN::CPU::query(std::string &brand, TRN::CPU::Implementation &implementation)
{
	int nIds_;
	int nExIds_;
	std::string vendor_;
	std::string brand_;
	bool isIntel_;
	bool isAMD_;
	std::bitset<32> f_1_ECX_;
	std::bitset<32> f_1_EDX_;
	std::bitset<32> f_7_EBX_;
	std::bitset<32> f_7_ECX_;
	std::bitset<32> f_81_ECX_;
	std::bitset<32> f_81_EDX_;
	std::vector<std::array<int, 4>> data_;
	std::vector<std::array<int, 4>> extdata_;

	//int cpuInfo[4] = {-1};  
	std::array<int, 4> cpui;
	int cpu_cores = boost::thread::physical_concurrency();
	

	// Calling __cpuid with 0x0 as the function_id argument  
	// gets the number of the highest valid function ID.  
	__cpuid(cpui.data(), 0);
	nIds_ = cpui[0];

	for (int i = 0; i <= nIds_; ++i)
	{
		__cpuidex(cpui.data(), i, 0);
		data_.push_back(cpui);
	}

	// Capture vendor string  
	char vendor_buf[0x20];
	memset(vendor_buf, 0, sizeof(vendor_buf));
	*reinterpret_cast<int*>(vendor_buf) = data_[0][1];
	*reinterpret_cast<int*>(vendor_buf + 4) = data_[0][3];
	*reinterpret_cast<int*>(vendor_buf + 8) = data_[0][2];
	vendor_ = vendor_buf;

	if (vendor_ == "GenuineIntel")
	{
		isIntel_ = true;
	}
	else if (vendor_ == "AuthenticAMD")
	{
		isAMD_ = true;
	}

	// load bitset with flags for function 0x00000001  
	if (nIds_ >= 1)
	{
		f_1_ECX_ = data_[1][2];
		f_1_EDX_ = data_[1][3];
	}

	// load bitset with flags for function 0x00000007  
	if (nIds_ >= 7)
	{
		f_7_EBX_ = data_[7][1];
		f_7_ECX_ = data_[7][2];
	}

	// Calling __cpuid with 0x80000000 as the function_id argument  
	// gets the number of the highest valid extended ID.  
	__cpuid(cpui.data(), 0x80000000);
	nExIds_ = cpui[0];

	char brand_buf[0x40];
	memset(brand_buf, 0, sizeof(brand_buf));

	for (int i = 0x80000000; i <= nExIds_; ++i)
	{
		__cpuidex(cpui.data(), i, 0);
		extdata_.push_back(cpui);
	}

	// load bitset with flags for function 0x80000001  
	if (nExIds_ >= 0x80000001)
	{
		f_81_ECX_ = extdata_[1][2];
		f_81_EDX_ = extdata_[1][3];
	}

	// Interpret CPU brand string if reported  
	if (nExIds_ >= 0x80000004)
	{
		memcpy(brand_buf, extdata_[2].data(), sizeof(cpui));
		memcpy(brand_buf + 16, extdata_[3].data(), sizeof(cpui));
		memcpy(brand_buf + 32, extdata_[4].data(), sizeof(cpui));
		brand_ = brand_buf;
	}
	bool sse = f_1_EDX_[25];
	bool sse2 = f_1_EDX_[26];
	bool sse3 = f_1_ECX_[0];
	bool sse41 = f_1_ECX_[19];
	bool avx = f_1_ECX_[28];
	bool avx2 = f_7_EBX_[5];
	bool mmx = f_1_EDX_[23];
	bool fma3 = f_1_ECX_[12];

#if defined(_M_IX86) && !defined(_M_X64)
	if (mmx && sse)
		return TRN::CPU::Implementation::MMX_SSE;
#endif
#if !defined(_M_IX86) && defined(_M_X64)

	std::size_t step;
	if (fma3 && avx2)
	{
		implementation = TRN::CPU::Implementation::AVX2_FMA3;
		step = TRN::CPU::Traits<TRN::CPU::Implementation::AVX2_FMA3>::step;
	}
	else if (avx)
	{
		implementation = TRN::CPU::Implementation::AVX;
		step = TRN::CPU::Traits<TRN::CPU::Implementation::AVX>::step;
	}
	else if (sse41)
	{
		implementation = TRN::CPU::Implementation::SSE41;
		step = TRN::CPU::Traits<TRN::CPU::Implementation::SSE41>::step;
	}
	else if (sse3)
	{
		implementation = TRN::CPU::Implementation::SSE3;
		step = TRN::CPU::Traits<TRN::CPU::Implementation::SSE3>::step;
	}
	else if (sse2)
	{
		implementation = TRN::CPU::Implementation::SSE2;
		step = TRN::CPU::Traits<TRN::CPU::Implementation::SSE2>::step;
	}
#endif
	else
	{
		implementation = TRN::CPU::Implementation::SCALAR;
		step = TRN::CPU::Traits<TRN::CPU::Implementation::SCALAR>::step;
	}
	brand = brand_buf;



	std::string::iterator new_end = std::unique(brand.begin(), brand.end(), [](char lhs, char rhs) { return (lhs == rhs) && (lhs == ' '); });
	brand.erase(new_end, brand.end());

	auto idx = std::find(brand.begin(), brand.end(), '@');
	auto freq_str = brand.substr(std::distance(brand.begin(), idx) + 1);
	float freq_ghz;
	std::stringstream(freq_str) >> freq_ghz;
	
	auto simd_cores = step * cpu_cores;

	auto glfops = 2* freq_ghz * simd_cores;

	std::stringstream stream;
	stream << brand << " / " << simd_cores << " simd cores (" << std::fixed << std::setprecision(2)<<  glfops << " GFLOPS/s)";
	brand = stream.str();


}
