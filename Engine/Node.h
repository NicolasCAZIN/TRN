#pragma once

#include "engine_global.h"
#include "Communicator.h"
#include "Task.h"
#include "Helper/Bridge.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Node : public virtual TRN::Engine::Task, 
									public TRN::Helper::Bridge<TRN::Engine::Communicator, std::weak_ptr>
		{
		protected:
			class Handle;
			std::unique_ptr<Handle> handle;

		protected :
			Node(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank);

		public:
			virtual ~Node();

	
		public :
			void dispose();
		private:
			void body() override;
			void erase_functors(const unsigned long long &id);


		protected:
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::QUIT> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::START> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::STOP> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::ALLOCATE> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATE> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::TRAIN> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::TEST> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SEQUENCE> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SET> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_STATES> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_WEIGHTS> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_PERFORMANCES> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_SCHEDULING> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_BEGIN> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_END> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_UNIFORM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_GAUSSIAN> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_CUSTOM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::POSITION> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::STIMULUS> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SCHEDULING> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDBACK_WEIGHTS> &message) = 0;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::READOUT_WEIGHTS> &message) = 0;
		};

	};
};
