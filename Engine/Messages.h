#pragma once

#include "engine_global.h"



namespace TRN
{
	namespace Engine
	{



		union Identifier
		{
			struct
			{
				unsigned long long simulation_number : 32;
				unsigned long long condition_number : 16;
				unsigned long long number : 16;
			};

			unsigned long long id;
		};

		unsigned int ENGINE_EXPORT checksum(const std::vector<float> &sequence);

		void  ENGINE_EXPORT encode(const unsigned short &number, const unsigned short &condition_number, const unsigned int &simulation_number, unsigned long long &id);
		void ENGINE_EXPORT decode(const unsigned long long &id, unsigned short &number, unsigned short &condition_number, unsigned int &simulation_number);

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
			CONFIGURE_FEEDBACK_UNIFORM,
			CONFIGURE_FEEDBACK_GAUSSIAN,
			CONFIGURE_FEEDBACK_CUSTOM,
			CONFIGURE_READOUT_UNIFORM,
			CONFIGURE_READOUT_GAUSSIAN,
			CONFIGURE_READOUT_CUSTOM,
			/* simulation / worker -> client */
			POSITION,
			STIMULUS,
			SCHEDULING,
			FEEDFORWARD_WEIGHTS,
			RECURRENT_WEIGHTS,
			FEEDBACK_WEIGHTS,
			READOUT_WEIGHTS,
			MUTATOR_CUSTOM,
			SCHEDULER_CUSTOM,
			FEEDFORWARD_DIMENSIONS,
			RECURRENT_DIMENSIONS,
			FEEDBACK_DIMENSIONS,
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
		};

		struct Simulation : public FromFrontend
		{
			unsigned long long id;
			size_t counter;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<FromFrontend>(*this);
				ar & id;
				ar & counter;
			}
		};

		struct FromBackend
		{
			int rank;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & rank;
			}
		};



		template <enum TRN::Engine::Tag>
		struct Message
		{
		};

		template <>
		struct Message<TRN::Engine::Tag::START> : public FromFrontend
		{

		};

		template <>
		struct Message<TRN::Engine::Tag::STOP> : public FromFrontend
		{
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
		};

		template <>
		struct Message<TRN::Engine::Tag::TERMINATED> : public FromBackend
		{
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<FromBackend>(*this);
			}
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
		};

		template <>
		struct Message<TRN::Engine::Tag::CONFIGURED> : public Simulation
		{
		};
		template <>
		struct Message<TRN::Engine::Tag::ALLOCATED> : public Simulation
		{

		};
		template <>
		struct Message<TRN::Engine::Tag::DEALLOCATED> : public Simulation
		{

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
		};

		template<>
		struct Message<TRN::Engine::Tag::ALLOCATE> : public Simulation
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::DEALLOCATE> : public Simulation
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::TRAIN> : public Simulation
		{
			std::string label;
			std::string incoming;
			std::string expected;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & label;
				ar & incoming;
				ar & expected;
			}
		};
	
		template<>
		struct Message<TRN::Engine::Tag::TEST> : public Simulation
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
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & label;
				ar & incoming;
				ar & expected;
				ar & preamble;
				ar & autonomous;
				ar & supplementary_generations;
			}

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
		};

		template<>
		struct Message<TRN::Engine::Tag::SETUP_STATES> : public Setup
		{
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
		};

		template<>
		struct Message<TRN::Engine::Tag::SETUP_PERFORMANCES> : public Setup
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::SETUP_SCHEDULING> : public Simulation
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_BEGIN> : public Simulation
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_END> : public Simulation
		{
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
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> : public ConfigureMeasurement
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> : public ConfigureMeasurement
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> : public ConfigureMeasurement
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> : public ConfigureMeasurement
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> : public ConfigureMeasurement
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> : public ConfigureMeasurement
		{
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
				ar & seed;
			}
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
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> : public Loop
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> : public Loop
		{
			std::size_t rows;
			std::size_t cols;
			std::pair<float, float> x;
			std::pair<float, float> y;
			float sigma;
			float radius;
			float scale;
			std::string tag;
			unsigned long seed;
			unsigned int checksum;
			std::vector<float> sequence;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Loop>(*this);
				ar & checksum;
				ar & sequence;
				ar & rows;
				ar & cols;
				ar & x;
				ar & y;
				ar & sigma;
				ar & radius;
				ar & scale;
				ar & tag;
				ar & seed;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> : public Loop
		{
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
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> : public Simulation
		{
			unsigned long seed;
			unsigned int snippets_size;
			unsigned int time_budget;
			std::string tag;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & snippets_size;
				ar & time_budget;
				ar & seed;
				ar & tag;
			}
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
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> : public Uniform
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> : public Gaussian
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> : public Simulation
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> : public Uniform
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> : public Gaussian
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> : public Simulation
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_UNIFORM> : public Uniform
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_GAUSSIAN> : public Gaussian
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_CUSTOM> : public Simulation
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> : public Uniform
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> : public Gaussian
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> : public Simulation
		{
		};
		struct Dimensions : public Simulation
		{
			std::size_t matrices;
			std::size_t rows;
			std::size_t cols;
			unsigned long seed;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & matrices;
				ar & rows;
				ar & cols;
				ar & seed;
			}
		};


		struct Matrix : public Dimensions
		{
			std::vector<float> elements;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Dimensions>(*this);
				ar & elements;
			}
		};

		struct Measurement : public Matrix
		{
			std::vector<float> expected;
			std::vector<float> primed;
			std::size_t preamble;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Matrix>(*this);
				ar & expected;
				ar & primed;
				ar & preamble;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::POSITION> : public Matrix
		{
			std::size_t trial;
			std::size_t evaluation;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Matrix>(*this);
				ar & trial;
				ar & evaluation;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::STIMULUS> : public Message<TRN::Engine::Tag::POSITION>
		{
	
		};

		template<>
		struct Message<TRN::Engine::Tag::SCHEDULING> : public Simulation
		{
			std::vector<int> offsets;
			std::vector<int> durations;
			bool is_from_mutator;
			std::size_t trial;
	

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & offsets;
				ar & durations;
				ar & is_from_mutator;
				ar & trial;
		
			}
		};


		template<>
		struct Message<TRN::Engine::Tag::MUTATOR_CUSTOM> : public Simulation
		{
			std::size_t trial;
			std::vector<int> offsets;
			std::vector<int> durations;
			unsigned long seed;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & trial;
				ar & offsets;
				ar & durations;
				ar & seed;
			}
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
		};


		template<>
		struct Message<TRN::Engine::Tag::FEEDBACK_DIMENSIONS> : public Dimensions
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::FEEDFORWARD_DIMENSIONS> : public Dimensions
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::RECURRENT_DIMENSIONS> : public Dimensions
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::READOUT_DIMENSIONS> : public Dimensions
		{
		};


		template<>
		struct Message<TRN::Engine::Tag::FEEDBACK_WEIGHTS> : public Matrix
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> : public Matrix
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> : public Matrix
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::READOUT_WEIGHTS> : public Matrix
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::WEIGHTS> : public Matrix
		{
			std::size_t trial;
			std::size_t batch;
			std::string label;
			std::string phase;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Matrix>(*this);
				ar & label;
				ar & phase;
				ar & batch;
				ar & trial;
			}
		};
		template<>
		struct Message<TRN::Engine::Tag::STATES> : public Message<TRN::Engine::Tag::WEIGHTS>
		{
			std::size_t evaluation;
		
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Message<TRN::Engine::Tag::WEIGHTS>>(*this);
				ar & evaluation;
			}
		};

	

		template<>
		struct Message<TRN::Engine::Tag::PERFORMANCES> : public Simulation
		{
			std::size_t trial;
			std::size_t evaluation;
			float cycles_per_second;
			float gflops_per_second;
			std::string phase;
	
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Simulation>(*this);
				ar & trial;
				ar & evaluation;
				ar & cycles_per_second;
				ar & gflops_per_second;
				ar & phase;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::TRAINED> : public Simulation
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::TESTED> : public Simulation
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::PRIMED> : public Simulation
		{
		};
	
		
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> : public Matrix
		{
			std::size_t trial;
			std::size_t evaluation;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Matrix>(*this);
				ar & trial;
				ar & evaluation;
			}
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_READOUT_FRECHET_DISTANCE> : public Matrix
		{
			std::size_t trial;
			std::size_t evaluation;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Matrix>(*this);
				ar & trial;
				ar & evaluation;
			}
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_READOUT_CUSTOM> : public Measurement
		{
			std::size_t trial;
			std::size_t evaluation;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Measurement>(*this);
				ar & trial;
				ar & evaluation;
			}
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> : public Matrix
		{
			std::size_t trial;
			std::size_t evaluation;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Matrix>(*this);
				ar & trial;
				ar & evaluation;
			}
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_POSITION_FRECHET_DISTANCE> : public Matrix
		{
			std::size_t trial;
			std::size_t evaluation;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Matrix>(*this);
				ar & trial;
				ar & evaluation;
			}
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_POSITION_CUSTOM> : public Measurement
		{
			std::size_t trial;
			std::size_t evaluation;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Measurement>(*this);
				ar & trial;
				ar & evaluation;
			}
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
		};

		template<>
		struct Message<TRN::Engine::Tag::LOG_WARNING> : public Message<TRN::Engine::Tag::LOG_INFORMATION>
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::LOG_ERROR> : public Message<TRN::Engine::Tag::LOG_INFORMATION>
		{
		};


	};
};

ENGINE_EXPORT std::ostream  & operator << (std::ostream &os, const TRN::Engine::Tag &tag);