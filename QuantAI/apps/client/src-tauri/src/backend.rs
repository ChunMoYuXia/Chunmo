use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Serialize)]
pub struct Connection {
    pub url: String,
    pub token: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Handshake {
    version: u8,
    url: String,
}

pub struct OwnedBackend {
    child: Child,
    connection: Connection,
}

impl OwnedBackend {
    pub fn start(executable: &Path, data_dir: &Path) -> Result<Self, String> {
        fs::create_dir_all(data_dir).map_err(|_| "无法创建本地数据目录")?;
        let log = File::create(data_dir.join("backend.log")).map_err(|_| "无法创建本地诊断日志")?;
        let mut random = [0u8; 32];
        OsRng.fill_bytes(&mut random);
        let token: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
        let mut command = Command::new(executable);
        command
            .args(["--sidecar", "--parent-stdin", "--port", "0", "--data-dir"])
            .arg(data_dir)
            .env("PORTFOLIO_SESSION_TOKEN", &token)
            .env("PYTHONUTF8", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log));
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        let mut child = command
            .spawn()
            .map_err(|_| "无法启动随附研究服务，请检查安装完整性")?;
        let stdout = child.stdout.take().ok_or("研究服务缺少握手通道")?;
        // Establish ownership before any fallible readiness operation.
        let mut owned = Self {
            child,
            connection: Connection {
                url: String::new(),
                token,
            },
        };
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let mut line = String::new();
            let result = BufReader::new(stdout)
                .take(4096)
                .read_line(&mut line)
                .map(|_| line);
            let _ = sender.send(result);
        });
        let line = receiver
            .recv_timeout(Duration::from_secs(30))
            .map_err(|_| "研究服务握手超时，请检查本地诊断日志")?
            .map_err(|_| "无法读取研究服务握手")?;
        let handshake: Handshake =
            serde_json::from_str(&line).map_err(|_| "研究服务握手格式无效")?;
        let port: u16 = handshake
            .url
            .strip_prefix("http://127.0.0.1:")
            .ok_or("研究服务地址必须为本机")?
            .parse()
            .map_err(|_| "研究服务端口无效")?;
        if handshake.version != 1 || port == 0 {
            return Err("研究服务握手版本或端口不兼容".into());
        }
        owned.connection.url = handshake.url;
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if owned
                .child
                .try_wait()
                .map_err(|_| "无法检查研究服务状态")?
                .is_some()
            {
                return Err("研究服务提前退出，请检查本地诊断日志".into());
            }
            if authenticated_health(port, &owned.connection.token) {
                return Ok(owned);
            }
            thread::sleep(Duration::from_millis(100));
        }
        Err("研究服务健康检查失败，请检查本地诊断日志".into())
    }

    pub fn connection(&mut self) -> Result<Connection, String> {
        match self.child.try_wait() {
            Ok(None) => Ok(self.connection.clone()),
            _ => Err("本地研究服务已经退出，请重新连接".into()),
        }
    }
}

impl Drop for OwnedBackend {
    fn drop(&mut self) {
        // EOF requests graceful shutdown; only this Child is ever terminated.
        self.child.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(12);
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn authenticated_health(port: u16, token: &str) -> bool {
    let address: SocketAddr = ([127, 0, 0, 1], port).into();
    let Ok(mut socket) = TcpStream::connect_timeout(&address, Duration::from_millis(500)) else {
        return false;
    };
    let _ = socket.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = socket.set_write_timeout(Some(Duration::from_millis(500)));
    let request = format!("GET /v1/config HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n");
    if socket.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    if socket.take(32768).read_to_string(&mut response).is_err() {
        return false;
    }
    if !response.starts_with("HTTP/1.1 200 ") {
        return false;
    }
    let Some((_, body)) = response.split_once("\r\n\r\n") else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    value["data"]["mode"] == "local" && value["data"]["version"] == "0.1.0"
}
