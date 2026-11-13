//! Rust desktop shell: the Electron agent logic over HTTP, serving the web
//! renderer and driving a bundled Chromium. Replaces fennec-desktop/electron/*.js.
pub mod config;
pub mod engine;