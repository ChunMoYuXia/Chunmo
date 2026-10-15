/** A response must still own the current scope, even when a transport ignores abort. */
export class RequestScope {
  private controller: AbortController | null = null;
  private revision = 0;

  start(): { signal: AbortSignal; current: () => boolean } {
    this.cancel();
    const controller = new AbortController();
    this.controller = controller;
    const revision = this.revision;
    return {
      signal: controller.signal,
      current: () => revision === this.revision && !controller.signal.aborted,
    };
  }

  cancel(): void {
    this.controller?.abort();
    this.revision += 1;
  }
}
