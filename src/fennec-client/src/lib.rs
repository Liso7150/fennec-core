//! Client for the Fennec loopback services, owned by fennec-core because the
//! core owns both contracts: `server.py` (the agent server) and
//! `engine.py` (the reference client for the Fennec engine).
//! Native Rust shells and editor integrations consume this instead of
//! re-implementing routes, headers, SSE framing or cancellation.
//! std-only: no third-party HTTP, so offline builds keep working.
pub mod agent;
pub mod coding_agent;
pub mod engine;
pub mod git;
pub mod http;
pub mod suite;

pub use agent::{Core, Outlet, Outlets, outlet_rows, OUTLET_RUN_TIMEOUT};
pub use coding_agent::{
    AgentTask, BackendCapabilities, CodingAgentClient, MessageContent, SseEvent, TaskMessage,
};
pub use engine::{ChatDelta, Engine};
pub use suite::{Repo, Suite};
