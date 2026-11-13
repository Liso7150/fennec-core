//! Client for the Fennec loopback services, owned by fennec-core because the
//! core owns both contracts: `fennec_core/server.py` (the agent server) and
//! `fennec_core/engine.py` (the reference client for the Fennec engine).
//! Native shells — the desktop, an editor plugin, a Node binding — consume this
//! instead of re-implementing routes, headers, SSE framing or cancellation.
//! std-only: no third-party HTTP, so offline builds keep working.
pub mod agent;
pub mod engine;
pub mod git;
pub mod http;
pub mod suite;

pub use agent::Core;
pub use engine::{ChatDelta, Engine};
pub use suite::{Repo, Suite};
