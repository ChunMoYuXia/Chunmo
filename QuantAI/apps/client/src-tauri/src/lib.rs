mod backend;

use backend::{Connection, OwnedBackend};
use std::sync::{Arc, Mutex};
use tauri::{Manager, State, WebviewUrl, WebviewWindow, WebviewWindowBuilder};

#[derive(Default)]
struct BackendState(Arc<Mutex<Option<OwnedBackend>>>);

fn trusted_origin(scheme: &str, host: Option<&str>, port: Option<u16>) -> bool {
    (scheme == "tauri" && host == Some("localhost"))
        || (scheme == "http" && host == Some("tauri.localhost") && port.is_none())
        || (cfg!(debug_assertions)
            && scheme == "http"
            && host == Some("127.0.0.1")
            && port == Some(5173))
}

#[tauri::command]
async fn backend_connection(
    app: tauri::AppHandle,
    window: WebviewWindow,
    state: State<'_, BackendState>,
) -> Result<Connection, String> {
    let url = window.url().map_err(|_| "无法检查客户端来源")?;
    if window.label() != "main" || !trusted_origin(url.scheme(), url.host_str(), url.port()) {
        return Err("客户端来源不受信任".into());
    }
    let resources = if cfg!(debug_assertions) {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    } else {
        app.path()
            .resource_dir()
            .map_err(|_| "无法定位随附研究服务")?
    };
    let executable = resources.join("resources/backend/portfolio-api.exe");
    let data_dir = app
        .path()
        .app_local_data_dir()
        .map_err(|_| "无法定位本地数据目录")?;
    let shared = state.0.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<Connection, String> {
        let mut guard = shared.lock().map_err(|_| "本地服务状态异常")?;
        if let Some(backend) = guard.as_mut() {
            if let Ok(connection) = backend.connection() {
                return Ok(connection);
            }
        }
        guard.take();
        let mut backend = OwnedBackend::start(&executable, &data_dir)?;
        let connection = backend.connection()?;
        *guard = Some(backend);
        Ok(connection)
    })
    .await
    .map_err(|_| "研究服务初始化异常".to_string())?
}

pub fn run() {
    let app = tauri::Builder::default()
        .manage(BackendState::default())
        .invoke_handler(tauri::generate_handler![backend_connection])
        .setup(|app| {
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("组合研究台")
                .inner_size(1440.0, 900.0)
                .min_inner_size(1000.0, 680.0)
                .on_navigation(|url| trusted_origin(url.scheme(), url.host_str(), url.port()))
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("无法初始化桌面客户端");
    app.run(|app, event| {
        if matches!(event, tauri::RunEvent::Exit) {
            if let Ok(mut backend) = app.state::<BackendState>().0.lock() {
                backend.take();
            }
        }
    });
}
