#pragma once

#include "Node.h"
#include "Broker.h"
#include "Helper/Visitor.h"
namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Proxy : 
			public std::enable_shared_from_this<TRN::Engine::Proxy>,
			public TRN::Engine::Node
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Proxy(const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy, const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor);
			virtual ~Proxy();

		protected :
			virtual void uninitialize() override;
			virtual void initialize() override;
	
			

		public:
			//virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::READY> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::COMPLETED> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::ALLOCATE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::TRAIN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::TEST> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SEQUENCE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SET> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_STATES> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_WEIGHTS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_PERFORMANCES> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_SCHEDULING> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_BEGIN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_END> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_UNIFORM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_GAUSSIAN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::POSITION> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::STIMULUS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SCHEDULING> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDBACK_WEIGHTS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::READOUT_WEIGHTS> &message) override;

			static std::shared_ptr<Proxy> create(const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy, const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor);
		};
	};
};

