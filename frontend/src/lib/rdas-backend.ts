// BackendInterface (simulated) — mirrors the C++ BackendInterface class
// that would receive frames + inference from a YOLO backend.

import type { Alarm, AlarmLevel } from "@/components/rdas/AlarmPanel";
import type { DriverSignal } from "@/components/rdas/DriverStatusPanel";
import type { LogEvent } from "@/components/rdas/EventLogWidget";

const priority: Record<AlarmLevel, number> = { normal: 0, warning: 1, high: 2, critical: 3 };

export function highestAlarm(alarms: Alarm[]): Alarm {
  return alarms.reduce((a, b) => (priority[b.level] > priority[a.level] ? b : a));
}

export const initialSignals: DriverSignal[] = [
  { key: "present", label: "Driver Present", active: true, level: "normal", value: "detected · 12m 04s" },
  { key: "drowsy", label: "Drowsiness", active: true, level: "warning", value: "PERCLOS 0.28" },
  { key: "yawn", label: "Yawning", active: false, level: "off" },
  { key: "phone", label: "Phone Usage", active: true, level: "high", value: "device in hand" },
  { key: "smoke", label: "Smoking", active: false, level: "off" },
  { key: "headphones", label: "Headphones", active: false, level: "off" },
  { key: "distracted", label: "Distracted Driver", active: true, level: "warning", value: "gaze off-road 3.2s" },
  { key: "covered", label: "Camera Covered", active: false, level: "off" },
];

export const initialAlarms: Alarm[] = [
  { code: "A-102", title: "Phone Usage Detected", detail: "Driver holding mobile device — sustained > 2s", level: "high", since: "00:00:04" },
  { code: "A-051", title: "Drowsiness Warning", detail: "Eye closure ratio elevated", level: "warning", since: "00:00:11" },
];

export const initialEvents: LogEvent[] = [
  { id: "1", ts: "12:04:22.184", event: "phone_detected", confidence: 0.62, level: "high" },
  { id: "2", ts: "12:04:19.902", event: "gaze_off_road", confidence: 0.71, level: "warning" },
  { id: "3", ts: "12:04:11.550", event: "eye_closure_high", confidence: 0.68, level: "warning" },
  { id: "4", ts: "12:03:58.220", event: "driver_present", confidence: 0.97, level: "normal" },
  { id: "5", ts: "12:03:56.010", event: "camera_stream_online", confidence: 1.0, level: "normal" },
  { id: "6", ts: "12:03:55.900", event: "yolo_model_loaded", confidence: 1.0, level: "normal" },
  { id: "7", ts: "12:03:55.812", event: "backend_handshake_ok", confidence: 1.0, level: "normal" },
];
