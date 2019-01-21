#pragma once

#include "trn4cpp_global.h"

#include "Basic.h"

namespace TRN4CPP
{
	namespace Plugin
	{
		namespace Sequences
		{
			class TRN4CPP_EXPORT Interface : public Plugin::Basic::Interface
			{
			public:
				virtual void callback_variable(const std::string &label, const std::string &tag) = 0;
				virtual void install_variable(const std::function<void(const std::string &label, const std::string &tag, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)> &functor) = 0;
			};

			void TRN4CPP_EXPORT		initialize(const std::string &library_path, const std::string &name, const std::map<std::string, std::string>  &arguments);
			void TRN4CPP_EXPORT uninitialize();
		};
	};

	namespace Sequences
	{
		extern const std::string DEFAULT_TAG;

		void TRN4CPP_EXPORT  	declare(const std::string &label, const std::string &tag, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols);

/* internal */
		void TRN4CPP_EXPORT  	retrieve(const std::string &label, const std::string &tag, std::vector<float> &elements, std::size_t &rows, std::size_t &cols);
		void TRN4CPP_EXPORT		fetch(const std::string &label, const std::string &tag);

	};
};



