#pragma once

#include "Scheduling.h"
#include "Batch.h"

namespace TRN
{
	namespace Core
	{
		namespace Message
		{
			enum  Type
			{
				STIMULUS,
				PREDICTION,
				POSITION,
				PERFORMANCES,
				TARGET_TRAJECTORY,
				TARGET_SEQUENCE,
				PARAMETERS,
				STATES,
				WEIGHTS,
				PRIMED,
				TESTED,
				TRAINED,
				CONFIGURED,
				CYCLES,
				FLOPS,
				SCHEDULING,
				SET,
				TEST
			};

			template <TRN::Core::Message::Type Type>
			class CORE_EXPORT Payload
			{
			};

			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::Type::STIMULUS>
			{
			private :
				class Handle;
				std::unique_ptr<Handle> handle;
			public :
				Payload(const std::shared_ptr<TRN::Core::Batch> &stimulus, const unsigned long long &evaluation_id);
				Payload(const  Payload<TRN::Core::Message::Type::STIMULUS> &payload);
				~Payload();
			public:
				const std::shared_ptr<TRN::Core::Batch> get_stimulus() const;
				const unsigned long long get_evaluation_id() const;
			};

			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::PREDICTION>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const std::shared_ptr<TRN::Core::Batch> &predicted, const unsigned long long &evaluation_id);
				Payload(const  Payload<TRN::Core::Message::Type::PREDICTION> &payload);
				~Payload();

			public :
				const std::shared_ptr<TRN::Core::Batch> get_predicted() const;
				const unsigned long long get_evaluation_id() const;
			};

			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::POSITION>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const std::shared_ptr<TRN::Core::Batch> &position, const unsigned long long &evaluation_id);
				Payload(const  Payload<TRN::Core::Message::Type::POSITION> &payload);
				~Payload();

			public:
				const std::shared_ptr<TRN::Core::Batch> get_position() const;
				const unsigned long long get_evaluation_id() const;
			};


			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::PERFORMANCES>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const float &cycles_per_second);
				Payload(const  Payload<TRN::Core::Message::Type::PERFORMANCES> &payload);
				~Payload();
			};
			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::TARGET_TRAJECTORY>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const std::shared_ptr<TRN::Core::Matrix> &trajectory);
				Payload(const  Payload<TRN::Core::Message::Type::TARGET_TRAJECTORY> &payload);
				~Payload();

			public:
				std::shared_ptr<TRN::Core::Matrix> get_trajectory() const;
			};
			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::TARGET_SEQUENCE>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const std::shared_ptr<TRN::Core::Matrix> &sequence);
				Payload(const  Payload<TRN::Core::Message::Type::TARGET_SEQUENCE> &payload);
				~Payload();

			public :
				std::shared_ptr<TRN::Core::Matrix> get_sequence() const;
			};
		
			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::STATES>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload();
				Payload(const  Payload<TRN::Core::Message::Type::STATES> &payload);
				~Payload();
			public :
				 std::size_t &get_rows() const;
				 std::shared_ptr<TRN::Core::Matrix> get_global() const;
				 std::shared_ptr<TRN::Core::Matrix> get_reservoir() const;
				 std::shared_ptr<TRN::Core::Matrix> get_prediction() const;
				 std::shared_ptr<TRN::Core::Matrix> get_desired() const;
				 std::shared_ptr<TRN::Core::Matrix> get_stimulus() const;
			public :
				void set_rows(const std::size_t &rows) const;
				void set_global(const std::shared_ptr<TRN::Core::Matrix> &global) const;
				void set_reservoir(const std::shared_ptr<TRN::Core::Matrix> &reservoir) const;
				void set_prediction(const std::shared_ptr<TRN::Core::Matrix> &prediction) const;
				void set_desired(const std::shared_ptr<TRN::Core::Matrix> &desired) const;
				void set_stimulus(const std::shared_ptr<TRN::Core::Matrix> &stimulus) const;
			public :
				static std::shared_ptr<Payload> create();
			};
			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::WEIGHTS>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload();
				Payload(const  Payload<TRN::Core::Message::Type::WEIGHTS> &payload);
				~Payload();
			public:
			 const std::shared_ptr<TRN::Core::Batch> get_feedforward() const;
			 const std::shared_ptr<TRN::Core::Batch> get_recurrent() const;
			 const std::shared_ptr<TRN::Core::Batch> get_readout() const;
			public :
			 void set_feedforward(const std::shared_ptr<TRN::Core::Batch> &feedforward) const;
			 void set_recurrent(const std::shared_ptr<TRN::Core::Batch> &recurrent) const;
			 void set_readout(const std::shared_ptr<TRN::Core::Batch> &readout) const;
			public:
				static std::shared_ptr<Payload> create();
			};

			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::TESTED>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const unsigned long long &evaluation_id);
				Payload(const  Payload<TRN::Core::Message::Type::TESTED> &payload);
				~Payload();

			public:
				const unsigned long long &get_evaluation_id() const;
			};
			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::PRIMED>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const unsigned long long &evaluation_id);
				Payload(const  Payload<TRN::Core::Message::Type::PRIMED> &payload);
				~Payload();

			public:
				const unsigned long long &get_evaluation_id() const;
			};
			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::TRAINED>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const unsigned long long &evaluation_id);
				Payload(const  Payload<TRN::Core::Message::Type::TRAINED> &payload);
				~Payload();

			public :
				const unsigned long long &get_evaluation_id() const;
			};
			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::CONFIGURED>
			{

			public:
				Payload();
				Payload(const  Payload<TRN::Core::Message::Type::CONFIGURED> &payload);
				~Payload();
			};
			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::Type::CYCLES>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const std::size_t &batch_size,  const std::size_t &cycles, const std::size_t &observations);
				Payload(const  Payload<TRN::Core::Message::Type::CYCLES> &payload);
				~Payload();
			public:
				const std::size_t &get_batch_size() const;
				const std::size_t &get_cycles() const;
				const std::size_t &get_observations() const;
			};
			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::Type::FLOPS>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const size_t &flops_per_epoch_factor, const size_t &flops_per_cycle);
				Payload(const  Payload<TRN::Core::Message::Type::FLOPS> &payload);
				~Payload();
			public:
				const size_t &get_flops_per_epoch_factor() const;
				const size_t &get_flops_per_cycle() const;

				void set_flops_per_epoch_factor(const size_t &flops_per_epoch_factor) const;
				void set_flops_per_cycle(const size_t &flops_per_cycle) const;
			};
			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::Type::SCHEDULING>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Scheduling> &scheduling);
				Payload(const  Payload<TRN::Core::Message::Type::SCHEDULING> &payload);
				~Payload();
			public:
				const std::shared_ptr<TRN::Core::Scheduling> get_scheduling() const;
				const unsigned long long get_evaluation_id() const;
			
			};

			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::Type::SET>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const std::string &label,
						const std::string &incoming,
						const std::string &expected,
						const 	unsigned long long  &evaluation_id);
				Payload(const Payload<TRN::Core::Message::Type::SET> &ref);
				~Payload();

			public:
				const std::string &get_label() const;
				const std::string &get_incoming() const;
				const std::string &get_expected() const;
				const unsigned long long &get_evaluation_id() const;
			};

			template <>
			class CORE_EXPORT Payload<TRN::Core::Message::Type::TEST>
			{
			private:
				class Handle;
				std::unique_ptr<Handle> handle;
			public:
				Payload(const std::string &label, const bool &autonomous, const std::size_t &preamble, const std::size_t &supplementary_generations);
				Payload(const Payload<TRN::Core::Message::Type::TEST> &ref);
				~Payload();

			public:
				const bool &get_autonomous() const;
				const std::string &get_label() const;
				const std::size_t &get_preamble() const;
				const std::size_t &get_supplementary_generations() const;
			};
		};
	};
};