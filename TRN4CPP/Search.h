#pragma once

#include "trn4cpp_global.h"
#include "Basic.h"

namespace TRN4CPP
{
	namespace Plugin
	{
		namespace Search
		{
			class TRN4CPP_EXPORT Interface : public Plugin::Basic::Interface
			{
			public:
				virtual void callback_generation(const unsigned short &condition_number, const std::vector<float> &score) = 0;
				virtual void install_generation(const std::function<void(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population)> &functor) = 0;
			};

			void TRN4CPP_EXPORT		initialize(const std::string &library_path, const std::string &name, const std::map<std::string, std::string>  &arguments);
			void uninitialize();
		};
	};

	namespace Search
	{
		//void TRN4CPP_EXPORT  	prepare(const std::size_t &rounds, const std::size_t &batch_number, const std::size_t &batch_size);
		unsigned int TRN4CPP_EXPORT		size(const unsigned short &condition_number);
		std::string TRN4CPP_EXPORT retrieve(const unsigned short &condition_number, const unsigned int &individual_number, const std::string &key);
		void TRN4CPP_EXPORT		populate(const unsigned short &condition_number, const std::vector<std::map<std::string, std::string>> &population);
		void TRN4CPP_EXPORT		evaluate(const unsigned long long &id, const std::vector<float> &score);
		void TRN4CPP_EXPORT		begin(const unsigned short &condition_number, const unsigned int &simulation_number, const std::size_t &batch_number, const std::size_t &batch_size);
		bool TRN4CPP_EXPORT		end(const unsigned short &condition_number);
		std::map<std::string, std::set<std::string>> TRN4CPP_EXPORT parse(const std::string &filename);
	};
};

