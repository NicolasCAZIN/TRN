#pragma once

#include "engine_global.h"



namespace TRN
{
	namespace Engine
	{



	

		unsigned int ENGINE_EXPORT checksum(const std::vector<float> &sequence);


		void ENGINE_EXPORT encode(const unsigned short &frontend_number, const unsigned short &condition_number, const unsigned int &batch_number, unsigned long long &simulation_id);
		void ENGINE_EXPORT decode(const unsigned long long &simulation_id, unsigned short &frontend_number, unsigned short &condition_number, unsigned int &batch_number);
		

		namespace Evaluation
		{
			void ENGINE_EXPORT encode(const unsigned short &trial_number, const unsigned short &train_number, const unsigned short &test_number, const unsigned short &repeat_number, unsigned long long &simulation_id);
			void ENGINE_EXPORT decode(const unsigned long long &simulation_id, unsigned short &trial_number, unsigned short &train_number, unsigned short &test_number, unsigned short &repeat_number);
		};

		enum Tag
		{
			INVALID = 0,
			QUIT,
			EXIT,
			TERMINATED,
			STOP,
			START,
			/*READY,*/
			CACHED,
			/* technical / worker -> client */
			WORKER,
			/* technical / client -> worker */
		
			/* simulation / client -> worker */
			ALLOCATE,
			DEALLOCATE,
			TRAIN,
			TEST,
			DECLARE_SEQUENCE,
			DECLARE_SET,

			SETUP_STATES,
			SETUP_WEIGHTS,
			SETUP_PERFORMANCES,
			SETUP_SCHEDULING,

			CONFIGURE_BEGIN,
			CONFIGURE_END,


			CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR,
			CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE,
			CONFIGURE_MEASUREMENT_READOUT_CUSTOM,
			CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR,
			CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE,
			CONFIGURE_MEASUREMENT_POSITION_CUSTOM,
			CONFIGURE_RESERVOIR_WIDROW_HOFF,
			CONFIGURE_ENCODER_MODEL,
			CONFIGURE_ENCODER_CUSTOM,
			CONFIGURE_DECODER_LINEAR,
			CONFIGURE_DECODER_KERNEL_MODEL,
			CONFIGURE_DECODER_KERNEL_MAP,
			CONFIGURE_LOOP_COPY,
			CONFIGURE_LOOP_SPATIAL_FILTER,
			CONFIGURE_LOOP_CUSTOM,
			CONFIGURE_SCHEDULER_TILED,
			CONFIGURE_SCHEDULER_SNIPPETS,
			CONFIGURE_SCHEDULER_CUSTOM,
			CONFIGURE_MUTATOR_SHUFFLE,
			CONFIGURE_MUTATOR_REVERSE,
			CONFIGURE_MUTATOR_PUNCH,
			CONFIGURE_MUTATOR_CUSTOM,
			CONFIGURE_FEEDFORWARD_UNIFORM,
			CONFIGURE_FEEDFORWARD_GAUSSIAN,
			CONFIGURE_FEEDFORWARD_CUSTOM,
			CONFIGURE_RECURRENT_UNIFORM,
			CONFIGURE_RECURRENT_GAUSSIAN,
			CONFIGURE_RECURRENT_CUSTOM,
			CONFIGURE_READOUT_UNIFORM,
			CONFIGURE_READOUT_GAUSSIAN,
			CONFIGURE_READOUT_CUSTOM,
			/* simulation / worker -> client */
			POSITION,
			STIMULUS,
			SCHEDULING,
			FEEDFORWARD_WEIGHTS,
			RECURRENT_WEIGHTS,
			READOUT_WEIGHTS,
			MUTATOR_CUSTOM,
			SCHEDULER_CUSTOM,
			FEEDFORWARD_DIMENSIONS,
			RECURRENT_DIMENSIONS,
			READOUT_DIMENSIONS,

			STATES,
			WEIGHTS,
			PERFORMANCES,
			CONFIGURED,

			TRAINED,
			TESTED,
			PRIMED,
			ALLOCATED,
			DEALLOCATED,
			MEASUREMENT_READOUT_MEAN_SQUARE_ERROR,
			MEASUREMENT_READOUT_FRECHET_DISTANCE,
			MEASUREMENT_READOUT_CUSTOM,
			MEASUREMENT_POSITION_MEAN_SQUARE_ERROR,
			MEASUREMENT_POSITION_FRECHET_DISTANCE,
			MEASUREMENT_POSITION_CUSTOM,
			LOG_INFORMATION,
			LOG_WARNING,
			LOG_ERROR
		};



		struct FromFrontend 
		{
			unsigned short number;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & number;
			}

			virtual ~FromFrontend() {}
		};

		struct Simulation
		{
			unsigned long long simulation_id;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & simulation_id;
			}

			virtual ~Simulation() {}
		};
		struct Result : public Simulation
		{
			unsigned long long evaluation_id;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & evaluation_id;
			}

			virtual ~Result() {}
		};

		struct FromBackend
		{
			int rank;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & rank;
			}

			virtual ~FromBackend() {}
		};



		template <enum TRN::Engine::Tag>
		struct Message
		{
			virtual ~Message() {}
		};

		template <>
		struct Message<TRN::Engine::Tag::START> : public FromFrontend
		{
			virtual ~Message() {}
		};

		template <>
		struct Message<TRN::Engine::Tag::STOP> : public FromFrontend
		{
			virtual ~Message() {}
		};

		template <>
		struct Message<TRN::Engine::Tag::EXIT> : public FromBackend
		{
			unsigned short number;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<FromBackend>(*this);
				ar & number;
			}
			virtual ~Message() {}
		};

		template <>
		struct Message<TRN::Engine::Tag::TERMINATED> : public FromBackend
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<FromBackend>(*this);
			}
			virtual ~Message() {}
		};
		template <>
		struct Message<TRN::Engine::Tag::CACHED> : public FromBackend
		{
			std::set<unsigned int> checksums;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<FromBackend>(*this);
				ar & checksums;
			}
			virtual ~Message() {}
		};
		template <>
		struct Message<TRN::Engine::Tag::WORKER> : public FromBackend
		{
			std::string host;
			std::string name;
			unsigned int index;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<FromBackend>(*this);
				ar & host;
				ar & name;
				ar & index;
			}
			virtual ~Message() {}
		};

		template <>
		struct Message<TRN::Engine::Tag::CONFIGURED> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};
		template <>
		struct Message<TRN::Engine::Tag::ALLOCATED> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};
		template <>
		struct Message<TRN::Engine::Tag::DEALLOCATED> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};
		/*template <>
		struct Message<TRN::Engine::Tag::READY> : public Header
		{
		};*/

	
		

		template <>
		struct Message<TRN::Engine::Tag::QUIT> : public FromFrontend
		{
			bool terminate;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<FromFrontend>(*this);
				ar & terminate;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::ALLOCATE> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::DEALLOCATE> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::TRAIN> : public Result
		{
			std::string label;
			std::string incoming;
			std::string expected;
			bool reset_readout;
			
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & label;
				ar & incoming;
				ar & expected;
				ar & reset_readout;
			}
			virtual ~Message() {}
		};
	
		template<>
		struct Message<TRN::Engine::Tag::TEST> : public Result
		{
			std::string label;
			std::string incoming;
			std::string expected;
			unsigned int preamble;
			bool autonomous;
			unsigned int supplementary_generations;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & label;
				ar & incoming;
				ar & expected;
				ar & preamble;
				ar & autonomous;
				ar & supplementary_generations;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::DECLARE_SEQUENCE> : public Simulation
		{
			std::string label;
			std::string tag;
			std::size_t observations;
			unsigned int checksum;
			std::vector<float> sequence;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & checksum;
				ar & sequence;
				ar & label;
				ar & tag;
				ar & observations;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::DECLARE_SET> : public Simulation
		{
			std::string label;
			std::string tag;
			std::vector<std::string> labels;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & label;
				ar & tag;
				ar & labels;
			}
			virtual ~Message() {}
		};

		struct Setup : public Simulation
		{
			bool train;
			bool prime;
			bool generate;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & train;
				ar & prime;
				ar & generate;
			}
			virtual ~Setup() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::SETUP_STATES> : public Setup
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Setup>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::SETUP_WEIGHTS> : public Simulation
		{
			bool train;
			bool initialization;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & train;
				ar & initialization;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::SETUP_PERFORMANCES> : public Setup
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Setup>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::SETUP_SCHEDULING> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_BEGIN> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_END> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};

		struct ConfigureMeasurement : public Simulation
		{
			std::size_t batch_size;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & batch_size;
			}
			virtual ~ConfigureMeasurement() {}
		};
		struct ConfigureFrechetDistance : public ConfigureMeasurement
		{
			std::string norm;
			std::string aggregator;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<ConfigureMeasurement>(*this);
				ar & norm;
				ar & aggregator;
			}
			virtual ~ConfigureFrechetDistance() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> : public ConfigureMeasurement
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<ConfigureMeasurement>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> : public ConfigureFrechetDistance
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<ConfigureFrechetDistance>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> : public ConfigureMeasurement
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<ConfigureMeasurement>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> : public ConfigureMeasurement
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<ConfigureMeasurement>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> : public ConfigureFrechetDistance
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<ConfigureFrechetDistance>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> : public ConfigureMeasurement
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<ConfigureMeasurement>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> : public Simulation
		{
			std::size_t stimulus_size;
			std::size_t prediction_size;
			std::size_t reservoir_size;
			float leak_rate;
			float initial_state_scale;
			float learning_rate;
			std::size_t batch_size;
			std::size_t mini_batch_size;
			unsigned long seed;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & stimulus_size;
				ar & prediction_size;
				ar & reservoir_size;
				ar & leak_rate;
				ar & initial_state_scale;
				ar & learning_rate;
				ar & batch_size;
				ar & mini_batch_size;
				ar & seed;
			}
			virtual ~Message() {}
		};


		struct Loop : public Simulation
		{
			std::size_t stimulus_size;
			std::size_t batch_size;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & stimulus_size;
				ar & batch_size;
			}
			virtual ~Loop() {}
		};


		struct Decoder : public Simulation
		{
			std::size_t stimulus_size;
			std::size_t batch_size;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & stimulus_size;
				ar & batch_size;
			}
			virtual ~Decoder() {}
		};

		struct Kernel : public Decoder
		{
			std::size_t rows;
			std::size_t cols;
			std::pair<float, float> x;
			std::pair<float, float> y;
			float sigma;
			float radius;
			float scale;
			float angle;
			unsigned long seed;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Decoder>(*this);
				ar & rows;
				ar & cols;
				ar & x;
				ar & y;
				ar & sigma;
				ar & radius;
				ar & angle;
				ar & scale;
				ar & seed;
			}
			virtual ~Kernel() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_DECODER_LINEAR> : public Decoder
		{
			std::vector<float> cx;
			std::vector<float> cy;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Decoder>(*this);
				ar & cx;
				ar & cy;
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_ENCODER_CUSTOM> : public Simulation
		{
			std::size_t batch_size;
			std::size_t stimulus_size;


			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & stimulus_size;
				ar & batch_size;

			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_ENCODER_MODEL> : public Message<TRN::Engine::Tag::CONFIGURE_ENCODER_CUSTOM>
		{

			std::vector<float> cx;
			std::vector<float> cy;
			std::vector<float> K;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Message<TRN::Engine::Tag::CONFIGURE_ENCODER_CUSTOM>>(*this);
				ar & stimulus_size;
				ar & batch_size;
				ar & cx;
				ar & cy;
				ar & K;
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MODEL> : public Kernel
		{
			std::vector<float> cx;
			std::vector<float> cy;
			std::vector<float> K;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Kernel>(*this);
				ar & cx;
				ar & cy;
				ar & K;
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MAP> : public Kernel
		{

			std::pair<unsigned int, std::vector<float>> response;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Kernel>(*this);
				ar & response;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> : public Loop
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Loop>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> : public Loop
		{
			std::string tag;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Loop>(*this);
				ar & tag;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> : public Loop
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Loop>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> : public Simulation
		{
			std::size_t epochs;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & epochs;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> : public Simulation
		{
			unsigned long seed;
			unsigned int snippets_size;
			unsigned int time_budget;
			float learn_reverse_rate;
			float generate_reverse_rate;
			float learning_rate;
			float discount;
			std::string tag;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & snippets_size;
				ar & time_budget;
				ar & seed;
				ar & learn_reverse_rate;
				ar & generate_reverse_rate;
				ar & learning_rate;
				ar & discount;
				ar & tag;
			}
			virtual ~Message() {}
		};



		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> : public Simulation
		{
			std::string tag;
			unsigned long seed;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & tag;
				ar & seed;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> : public Simulation
		{
			unsigned long seed;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & seed;
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> : public Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE>
		{
			std::size_t size;
			float rate;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE>>(*this);
				ar & size;
				ar & rate;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> : public Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE>
		{
			std::size_t repetition;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE>>(*this);
				ar & repetition;
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> : public Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE>
		{
		};

		struct Gaussian : public Simulation
		{
			float mu;
			float sigma;
			float sparsity;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & mu;
				ar & sigma;
				ar & sparsity;
			}
			virtual ~Gaussian() {}
		};

		struct Uniform : public Simulation
		{
			float a;
			float b;
			float sparsity;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & a;
				ar & b;
				ar & sparsity;
			}
			virtual ~Uniform() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> : public Uniform
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Uniform>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> : public Gaussian
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Gaussian>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> : public Uniform
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Uniform>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> : public Gaussian
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Gaussian>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};

		

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> : public Uniform
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Uniform>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> : public Gaussian
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Gaussian>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> : public Simulation
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
			}
			virtual ~Message() {}
		};

		struct Dimensions
		{
			std::size_t matrices;
			std::size_t rows;
			std::size_t cols;
			unsigned long seed;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & matrices;
				ar & rows;
				ar & cols;
				ar & seed;
			}
			virtual ~Dimensions() {}
		};

		struct Matrix
		{
			std::vector<float> elements;
			std::size_t rows;
			std::size_t cols;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
			
				ar & elements;
				ar & rows;
				ar & cols;
			}
			virtual ~Matrix() {}
		};
		struct MatrixBatch : public Matrix
		{
			std::size_t matrices;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Matrix>(*this);
				ar & matrices;
			
			}
			virtual ~MatrixBatch() {}
		};
		struct Measurement : public MatrixBatch
		{
			std::vector<float> expected;
			std::vector<float> primed;
			std::size_t preamble;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<MatrixBatch>(*this);
				ar & expected;
				ar & primed;
				ar & preamble;
			}
			virtual ~Measurement() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::POSITION> : public Result, public Matrix
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & boost::serialization::base_object<Matrix>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::STIMULUS> : public Message<TRN::Engine::Tag::POSITION>
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Message<TRN::Engine::Tag::POSITION>>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::SCHEDULING> : public Result
		{
			std::vector<int> offsets;
			std::vector<int> durations;
			bool is_from_mutator;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & offsets;
				ar & durations;
				ar & is_from_mutator;
			}
			virtual ~Message() {}
		};


		template<>
		struct Message<TRN::Engine::Tag::MUTATOR_CUSTOM> : public Result
		{
			std::vector<int> offsets;
			std::vector<int> durations;
			unsigned long seed;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & offsets;
				ar & durations;
				ar & seed;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::SCHEDULER_CUSTOM> : public Message<TRN::Engine::Tag::MUTATOR_CUSTOM>
		{
			std::vector<float> elements;
			std::size_t rows;
			std::size_t cols;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Message<TRN::Engine::Tag::MUTATOR_CUSTOM>>(*this);
		
				ar & elements;
				ar & rows;
				ar & cols;
			}
			virtual ~Message() {}
		};





		template<>
		struct Message<TRN::Engine::Tag::FEEDFORWARD_DIMENSIONS> : public Simulation, public Dimensions
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & boost::serialization::base_object<Dimensions>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::RECURRENT_DIMENSIONS> : public Simulation, public Dimensions
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & boost::serialization::base_object<Dimensions>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::READOUT_DIMENSIONS> : public Simulation, public Dimensions
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & boost::serialization::base_object<Dimensions>(*this);
			}
			virtual ~Message() {}
		};


		template<>
		struct Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> : public Simulation, public MatrixBatch
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & boost::serialization::base_object<Matrix>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> : public Simulation, public MatrixBatch
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & boost::serialization::base_object<Matrix>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::READOUT_WEIGHTS> : public Simulation, public MatrixBatch
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & boost::serialization::base_object<Matrix>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::WEIGHTS> : public Result, public MatrixBatch
		{
			std::size_t batch;
			std::string label;
			std::string phase;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Matrix>(*this);
				ar & boost::serialization::base_object<Result>(*this);
				ar & label;
				ar & phase;
				ar & batch;
		
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::STATES> : public Message<TRN::Engine::Tag::WEIGHTS>
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Message<TRN::Engine::Tag::WEIGHTS>>(*this);
			}
			virtual ~Message() {}
		};

	

		template<>
		struct Message<TRN::Engine::Tag::PERFORMANCES> : public Result
		{
			float cycles_per_second;
			float gflops_per_second;
			std::string phase;
	
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & cycles_per_second;
				ar & gflops_per_second;
				ar & phase;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::TRAINED> : public Result
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::TESTED> : public Result
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::PRIMED> : public Result
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
			}
			virtual ~Message() {}
		};
	
		
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> : public Result, public Matrix
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & boost::serialization::base_object<Matrix>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_READOUT_FRECHET_DISTANCE> : public Result, public Matrix
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & boost::serialization::base_object<Matrix>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_READOUT_CUSTOM> :  public Result, public Measurement
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & boost::serialization::base_object<Measurement>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> : public Result, public Matrix
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & boost::serialization::base_object<Matrix>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_POSITION_FRECHET_DISTANCE> : public Result, public Matrix
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & boost::serialization::base_object<Matrix>(*this);
			}
			virtual ~Message() {}
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_POSITION_CUSTOM> :public Result, public Measurement
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Result>(*this);
				ar & boost::serialization::base_object<Measurement>(*this);
			}
			virtual ~Message() {}
		};


		template<>
		struct Message<TRN::Engine::Tag::LOG_INFORMATION> : public Simulation
		{
			std::string message;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & message;
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::LOG_WARNING> : public Message<TRN::Engine::Tag::LOG_INFORMATION>
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Message<TRN::Engine::Tag::LOG_INFORMATION>>(*this);
			}
			virtual ~Message() {}
		};

		template<>
		struct Message<TRN::Engine::Tag::LOG_ERROR> : public Message<TRN::Engine::Tag::LOG_INFORMATION>
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Message<TRN::Engine::Tag::LOG_INFORMATION>>(*this);
			}
			virtual ~Message() {}
		};


	};
};

ENGINE_EXPORT std::ostream  & operator << (std::ostream &os, const TRN::Engine::Tag &tag);