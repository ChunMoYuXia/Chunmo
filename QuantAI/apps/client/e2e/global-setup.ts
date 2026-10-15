import { spawn } from 'node:child_process';
import { randomBytes, randomUUID } from 'node:crypto';
import { createInterface } from 'node:readline';
import { mkdir, writeFile, open } from 'node:fs/promises';
import { resolve } from 'node:path';
import http from 'node:http';

function health(url: string): Promise<boolean> {
  return new Promise((resolve) => {
    const request = http.get(url + '/health', { timeout: 1000 }, (response) => {
      response.resume();
      resolve(response.statusCode === 200);
    });
    request.on('error', () => resolve(false));
    request.on('timeout', () => {
      request.destroy();
      resolve(false);
    });
  });
}

export default async function globalSetup() {
  const root = resolve('../..');
  const directory = resolve(root, '.cache/e2e', randomUUID());
  await mkdir(directory, { recursive: true });
  const token = randomBytes(32).toString('hex');
  const log = await open(resolve(directory, 'api.log'), 'w');
  const command =
    process.env.PORTFOLIO_E2E_EXECUTABLE ??
    resolve(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const args = process.env.PORTFOLIO_E2E_EXECUTABLE ? [] : ['-m', 'portfolio_client'];
  const child = spawn(
    command,
    [
      ...args,
      '--sidecar',
      '--parent-stdin',
      '--port',
      '0',
      '--data-dir',
      resolve(directory, 'runtime'),
      ...(process.env.PORTFOLIO_E2E_EXECUTABLE
        ? []
        : ['--static-dir', resolve(root, 'apps/client/build')]),
    ],
    {
      cwd: root,
      windowsHide: true,
      stdio: ['pipe', 'pipe', log.fd],
      env: {
        ...process.env,
        PORTFOLIO_SESSION_TOKEN: token,
        PYTHONUTF8: '1',
        PORTFOLIO_AI_BASE_URL: '',
        PORTFOLIO_AI_MODEL: '',
        PORTFOLIO_AI_API_KEY: '',
      },
    },
  );
  await log.close();
  const stop = async () => {
    child.stdin?.end();
    if (child.exitCode !== null) return;
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        if (process.platform === 'win32' && child.pid) {
          const killer = spawn('taskkill', ['/PID', String(child.pid), '/T', '/F'], {
            windowsHide: true,
            stdio: 'ignore',
          });
          killer.on('exit', () => resolve());
        } else {
          child.kill('SIGKILL');
          resolve();
        }
      }, 12000);
      child.once('exit', () => {
        clearTimeout(timer);
        resolve();
      });
    });
  };
  try {
    const lines = createInterface({ input: child.stdout });
    const handshake = await new Promise<{ version: number; url: string }>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('后端握手超时')), 30000);
      lines.once('line', (line) => {
        clearTimeout(timer);
        try {
          resolve(JSON.parse(line));
        } catch {
          reject(new Error('后端握手格式错误'));
        }
      });
      child.once('error', reject);
      child.once('exit', (code) => {
        clearTimeout(timer);
        reject(new Error(`后端在握手前退出 (${code})`));
      });
    });
    if (handshake.version !== 1 || !/^http:\/\/127\.0\.0\.1:\d+$/.test(handshake.url))
      throw new Error('后端握手版本或地址不受支持');
    let ready = false;
    for (let i = 0; i < 100; i++) {
      if (await health(handshake.url)) {
        ready = true;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    if (!ready) throw new Error('后端健康检查失败');
    await mkdir(resolve(root, '.cache/e2e'), { recursive: true });
    await writeFile(
      resolve(root, '.cache/e2e/connection.json'),
      JSON.stringify({ url: handshake.url, token }),
      'utf8',
    );
    return stop;
  } catch (error) {
    await stop();
    throw error;
  }
}
