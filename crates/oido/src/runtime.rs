//! Runtime del pipeline: wrapper de las dos variantes (Batch/Streaming).
//! Encapsula `Pipeline` y `StreamingPipeline` para que el bin las use
//! sin saber cuál está activa.

use oido_core::{Pipeline, PipelineEvent};

#[derive(Debug)]
pub(crate) enum ActivePipeline {
    Batch(Pipeline),
    Streaming(oido_core::StreamingPipeline),
}

impl ActivePipeline {
    pub(crate) fn events(&self) -> crossbeam_channel::Receiver<PipelineEvent> {
        match self {
            ActivePipeline::Batch(p) => p.events(),
            ActivePipeline::Streaming(p) => p.events(),
        }
    }

    pub(crate) fn start(&mut self) -> anyhow::Result<()> {
        match self {
            ActivePipeline::Batch(p) => p.start(),
            ActivePipeline::Streaming(p) => p.start(),
        }
    }

    pub(crate) fn shutdown(&mut self) -> anyhow::Result<()> {
        match self {
            ActivePipeline::Batch(p) => p.shutdown(),
            ActivePipeline::Streaming(p) => p.shutdown(),
        }
    }
}
