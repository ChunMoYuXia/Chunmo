import {
  configureConnection,
  consumeSessionLink,
  request,
  browserBackendUrl,
  rememberBackendUrl,
  type BackendConnection,
} from './client';

let pendingBrowserConnection: BackendConnection | null = null;

export async function connectBackend(
  location: URL,
  cleanLocation: (url: string) => void,
  nativeConnection?: () => Promise<BackendConnection>,
): Promise<void> {
  if (nativeConnection) {
    configureConnection(await nativeConnection());
    await request('/v1/config');
    return;
  }
  const link = consumeSessionLink(location);
  if (link) {
    cleanLocation(link.cleanUrl);
    configureConnection(link.connection);
    pendingBrowserConnection = link.connection;
  }
  const active = pendingBrowserConnection ?? { url: browserBackendUrl(), token: '' };
  configureConnection(active);
  rememberBackendUrl(active.url);
  if (pendingBrowserConnection) {
    await request('/v1/session', { method: 'POST' });
    pendingBrowserConnection = null;
    configureConnection({ url: active.url, token: '' });
  }
  await request('/v1/config');
}
