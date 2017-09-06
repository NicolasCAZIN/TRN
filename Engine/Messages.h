#pragma once

#include "engine_global.h"

namespace TRN
{
	namespace Engine
	{
		enum Tag
		{
			/* technical / worker -> client */
			ACK,
			WORKER,
			/* technical / client -> worker */
			QUIT,
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
			SCHEDULING_REQUEST,
			FEEDFORWARD_DIMENSIONS,
			RECURRENT_DIMENSIONS,
			FEEDBACK_DIMENSIONS,
			READOUT_DIMENSIONS,
			STATES,
			WEIGHTS,
			PERFORMANCES,
			TRAINED,
			TESTED,
			PRIMED,
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

		struct Header
		{
			size_t number;
			unsigned int id;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & number;
				ar & id;
			}
		};

		template <enum TRN::Engine::Tag>
		struct Message
		{
		};
		template <>
		struct Message<TRN::Engine::Tag::QUIT> : public Header
		{
	
		};
		template <>
		struct Message<TRN::Engine::Tag::WORKER>
		{
			std::string host;
			std::string name;
			unsigned int index;
			int rank;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & host;
				ar & name;
				ar & index;
				ar & rank;
			}
		};
		template<>
		struct Message<TRN::Engine::Tag::ACK> : public Header
		{
			bool success;
			std::string cause;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & success;
				ar & cause;
			}
		};
		template<>
		struct Message<TRN::Engine::Tag::ALLOCATE> : public Header
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::DEALLOCATE> : public Header
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::TRAIN> : public Header
		{
			std::string label;
			std::string incoming;
			std::string expected;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & label;
				ar & incoming;
				ar & expected;
			}
		};
	
		template<>
		struct Message<TRN::Engine::Tag::TEST> : public Header
		{
			std::string label;
			std::string incoming;
			std::string expected;
			unsigned int preamble;
			unsigned int supplementary_generations;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & label;
				ar & incoming;
				ar & expected;
				ar & preamble;
				ar & supplementary_generations;
			}

		};

		template<>
		struct Message<TRN::Engine::Tag::DECLARE_SEQUENCE> : public Header
		{
			std::string label;
			std::string tag;
			std::vector<float> sequence;
			std::size_t observations;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & label;
				ar & tag;
				ar & sequence;
				ar & observations;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::DECLARE_SET> : public Header
		{
			std::string label;
			std::string tag;
			std::vector<std::string> labels;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & label;
				ar & tag;
				ar & labels;
			}
		};

		struct Setup : public Header
		{
			bool train;
			bool prime;
			bool generate;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
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
		struct Message<TRN::Engine::Tag::SETUP_WEIGHTS> : public Header
		{
			bool train;
			bool initialization;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & train;
				ar & initialization;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::SETUP_PERFORMANCES> : public Setup
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::SETUP_SCHEDULING> : public Header
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_BEGIN> : public Header
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_END> : public Header
		{
		};

		struct ConfigureMeasurement : public Header
		{
			std::size_t batch_size;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
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
		struct Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> : public Header
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
				ar & boost::serialization::base_object<Header>(*this);
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


		struct Loop : public Header
		{
			std::size_t stimulus_size;
			std::size_t batch_size;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
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
			std::vector<float> response;
			float sigma;
			float radius;
			float scale;
			std::string tag;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Loop>(*this);
				ar & rows;
				ar & cols;
				ar & x;
				ar & y;
				ar & response;
				ar & sigma;
				ar & radius;
				ar & scale;
				ar & tag;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> : public Loop
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> : public Header
		{
			std::size_t epochs;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & epochs;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> : public Header
		{
			unsigned int snippets_size;
			unsigned int time_budget;
			std::string tag;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & snippets_size;
				ar & time_budget;
				ar & tag;
			}
		};



		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> : public Header
		{
			std::string tag;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & tag;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> : public Header
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> : public Header
		{
			std::size_t size;
			float rate;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & size;
				ar & rate;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> : public Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE>
		{
			std::size_t number;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE>>(*this);
				ar & number;
			}
		};
		template<>
		struct Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> : public Header
		{
		};

		struct Gaussian : public Header
		{
			float mu;
			float sigma;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & mu;
				ar & sigma;
			}
		};

		struct Uniform : public Header
		{
			float a;
			float b;
			float sparsity;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
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
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> : public Header
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
		struct Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> : public Header
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
		struct Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_CUSTOM> : public Header
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
		struct Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> : public Header
		{
		};
		struct Dimensions : public Header
		{
			std::size_t matrices;
			std::size_t rows;
			std::size_t cols;
			unsigned long seed;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
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

		};

		template<>
		struct Message<TRN::Engine::Tag::STIMULUS> : public Matrix
		{
	
		};

		template<>
		struct Message<TRN::Engine::Tag::SCHEDULING> : public Header
		{
			std::vector<int> offsets;
			std::vector<int> durations;
			bool is_from_mutator;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & offsets;
				ar & durations;
				ar & is_from_mutator;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::SCHEDULING_REQUEST> : public Header
		{
			std::vector<float> elements;
			std::size_t rows;
			std::size_t cols;
			std::vector<int> offsets;
			std::vector<int> durations;


			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & elements;
				ar & rows;
				ar & cols;
				ar & offsets;
				ar & durations;
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
		struct Message<TRN::Engine::Tag::STATES> : public Matrix
		{
			std::string label;
			std::string phase;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Matrix>(*this);
				ar & label;
				ar & phase;
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::WEIGHTS> : public Message<TRN::Engine::Tag::STATES>
		{
		};

		template<>
		struct Message<TRN::Engine::Tag::PERFORMANCES> : public Header
		{
			std::size_t cycles;
			std::size_t batch_size;
			float gflops;
			float seconds;
			std::string phase;
	
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
				ar & cycles;
				ar & batch_size;
				ar & gflops;
				ar & seconds;
				ar & phase;
	
			}
		};

		template<>
		struct Message<TRN::Engine::Tag::TRAINED> : public Header
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::TESTED> : public Header
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::PRIMED> : public Header
		{
		};
	
		
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> : public Matrix
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_READOUT_FRECHET_DISTANCE> : public Matrix
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_READOUT_CUSTOM> : public Measurement
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> : public Matrix
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_POSITION_FRECHET_DISTANCE> : public Matrix
		{
		};
		template<>
		struct Message<TRN::Engine::Tag::MEASUREMENT_POSITION_CUSTOM> : public Measurement
		{
		};


		template<>
		struct Message<TRN::Engine::Tag::LOG_INFORMATION> : public Header
		{
			std::string message;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & boost::serialization::base_object<Header>(*this);
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