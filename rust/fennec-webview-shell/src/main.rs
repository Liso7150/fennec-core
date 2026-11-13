//! WebView2 + native-IPC Rust shell — the token-stream winner configuration.
//! Rust agent backend (fennec-* crates) talking to the system WebView2 over
//! wry's IPC channel instead of SSE-over-loopback. Loads an inline bench page
//! that measures 2000-token delivery over IPC, for the transport A/B against
//! the Chromium+SSE shell. No bundled browser; uses the preinstalled runtime.
use std::sync::mpsc;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::window::WindowBuilder;
use wry::WebViewBuilder;

enum UserEvent {
    Stream(usize),
    Eval(String),
}

const BENCH_HTML: &str = r#"<!doctype html><html><head><meta charset="utf-8"></head>
<body><div id="root">webview ipc bench</div><script>
let n=0, t0=0, want=0;
window.__benchToken=function(i){ if(n===0)t0=performance.now(); n++;
  if(n>=want){ const ms=performance.now()-t0;
    window.ipc.postMessage(JSON.stringify({cmd:'record',name:'stream-delivered',elapsedMs:ms,count:n}));
    window.ipc.postMessage(JSON.stringify({cmd:'record',name:'stream-per-token',elapsedMs:ms/n,count:n}));
    document.getElementById('root').textContent='delivered '+n+' in '+ms.toFixed(1)+' ms';
  }};
window.ipc.postMessage(JSON.stringify({cmd:'stream_emit',count:2000}));
want=2000;
</script></body></html>"#;

fn record(name: &str, ms: f64, count: usize) {
    let out = std::env::var("SHELL_BENCH_OUT").unwrap_or_default();
    let line = serde_json::json!({"shell":"webview2-ipc","backend":name,"elapsed_ms":ms,"messages":count});
    if out.is_empty() {
        println!("{line}");
    } else if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&out) {
        use std::io::Write;
        let _ = writeln!(f, "{line}");
    }
}

fn main() {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    let window = WindowBuilder::new().with_title("Fennec WebView2 IPC").build(&event_loop).unwrap();
    let (tx, rx) = mpsc::channel::<String>();
    let ipc_proxy = proxy.clone();
    let webview = WebViewBuilder::new(&window)
        .with_html(BENCH_HTML)
        .with_ipc_handler(move |req| {
            let body = req.body().clone();
            let _ = tx.send(body);
        })
        .build()
        .unwrap();

    // Drain IPC on a background thread; dispatch stream/eval via the event loop.
    std::thread::spawn(move || {
        while let Ok(body) = rx.recv() {
            let v: serde_json::Value = match serde_json::from_str(&body) { Ok(v) => v, Err(_) => continue };
            match v.get("cmd").and_then(|c| c.as_str()).unwrap_or("") {
                "stream_emit" => {
                    let n = v.get("count").and_then(|c| c.as_u64()).unwrap_or(2000) as usize;
                    let _ = ipc_proxy.send_event(UserEvent::Stream(n));
                }
                "record" => {
                    let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
                    let ms = v.get("elapsedMs").and_then(|x| x.as_f64()).unwrap_or(0.0);
                    let count = v.get("count").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                    record(&name, ms, count);
                }
                _ => {}
            }
        }
    });

    let mut webview = Some(webview);
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => *control_flow = ControlFlow::Exit,
            Event::UserEvent(UserEvent::Stream(n)) => {
                if let Some(wv) = &webview {
                    let started = std::time::Instant::now();
                    for i in 0..n {
                        let _ = wv.evaluate_script(&format!("window.__benchToken({i})"));
                    }
                    let emit_ms = started.elapsed().as_secs_f64() * 1000.0;
                    record("stream-emit", emit_ms, n);
                }
            }
            Event::UserEvent(UserEvent::Eval(js)) => {
                if let Some(wv) = &webview { let _ = wv.evaluate_script(&js); }
            }
            _ => {}
        }
    });
}