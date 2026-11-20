/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** WebSocket URL for the RDAS backend's structured alarm/telemetry feed. Default: ws://localhost:8765 */
  readonly VITE_RDAS_WS_URL?: string;
  /** HLS URL for the MediaMTX-relayed annotated video stream. Default: http://localhost:8888/rdas/index.m3u8 */
  readonly VITE_RDAS_HLS_URL?: string;
  /** Alarm sound volume, 0..1. Default: 0.5 */
  readonly VITE_RDAS_ALARM_VOLUME?: string;
  /** ms to keep an alarm shown as acknowledged while the backend catches up. Default: 1500 */
  readonly VITE_RDAS_ACK_GRACE_MS?: string;
  /** ms an acknowledged alarm stays visible before being removed from the live views. Default: 5000 */
  readonly VITE_RDAS_ACKED_VISIBLE_MS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
