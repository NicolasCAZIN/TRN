#pragma once

#include "engine_global.h"
#include "Messages.h"
#include "Executor.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Processor : public TRN::Engine::Executor
		{
		public :
			enum Status
			{
				Deallocated,
				Allocating,
				Allocated,
				Configuring,
				Configured,
				//Ready,
				Training,
				Trained,
				Priming,
				Primed,
				Tested,
				Deallocating
			};

		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Processor( const int &rank);
			virtual ~Processor();

		public:
			std::string get_name();
			std::string get_host();
			int get_index();
			float get_latency();

	
			int get_rank();
	
			void set_host(const std::string &host);
			void set_name(const std::string &name);
			void set_index(const int &index);
			void set_t0(const clock_t &t0);
			void set_t1(const clock_t &t1);

			void allocating();
			void allocated();
			void configuring();
			void configured();
		//	void ready();
			void declare();
		
			void training();
			void trained();

			void primed();
			void tested();

			void testing();
			void deallocating();
			void deallocated();
		private :
			void wait(const std::function<bool(const TRN::Engine::Processor::Status &status)> &functor);
			void notify(const TRN::Engine::Processor::Status &status);
		public :
			static std::shared_ptr<TRN::Engine::Processor> create( const int &rank);
		};

		bool operator < (const std::shared_ptr<TRN::Engine::Processor> &left, const std::shared_ptr<TRN::Engine::Processor> &right);



	};
};
		


