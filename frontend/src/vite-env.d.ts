/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** WebSocket URL for the RDAS backend's structured alarm/telemetry feed. Default: ws://localhost:8765 */
  readonly VITE_RDAS_WS_URL?: string;
  /** HLS URL for the MediaMTX-relayed annotated video stream. Default: http://localhost:8888/rdas/index.m3u8 */
  readonly VITE_RDAS_HLS_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
