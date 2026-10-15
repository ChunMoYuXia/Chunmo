export function downloadText(filename: string, content: string, mediaType: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mediaType }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
