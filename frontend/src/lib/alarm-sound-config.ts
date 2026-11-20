// Central config for the acoustic alarm. Nothing here is a host/port; tune the
// sound by editing this file or by setting the VITE_* variables in .env.

import type { AlarmLevel } from "@/components/rdas/AlarmPanel";

export type SoundProfile = {
  /** Tone frequencies (Hz) played in order once per cycle. */
  freqs: number[];
  /** Length of each tone in milliseconds. */
  toneMs: number;
  /** Silence between tones within a cycle, in milliseconds. */
  gapMs: number;
  /** Time between the start of one cycle and the start of the next. */
  cycleMs: number;
  wave: OscillatorType;
};

function numEnv(raw: string | undefined, fallback: number, min: number, max: number): number {
  const n = raw === undefined || raw === "" ? NaN : Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

export const ALARM_SOUND_CONFIG = {
  /** Master volume, 0..1. Override with VITE_RDAS_ALARM_VOLUME. */
  volume: numEnv(import.meta.env.VITE_RDAS_ALARM_VOLUME as string | undefined, 0.5, 0, 1),
  /**
   * After the dispatcher hits ACK, keep treating the alarm as acknowledged for
   * this long even if a state frame already in flight still says acked=false.
   * Override with VITE_RDAS_ACK_GRACE_MS.
   */
  ackGraceMs: numEnv(import.meta.env.VITE_RDAS_ACK_GRACE_MS as string | undefined, 1500, 0, 10000),
  /**
   * How long an acknowledged alarm stays on screen before it is removed from the
   * live alarm views (it remains in the event log history). Override with
   * VITE_RDAS_ACKED_VISIBLE_MS.
   */
  ackedVisibleMs: numEnv(import.meta.env.VITE_RDAS_ACKED_VISIBLE_MS as string | undefined, 5000, 0, 600000),
  /** localStorage key used to remember the mute toggle. */
  muteStorageKey: "rdas.alarm.muted",
  /** Louder / faster patterns for more severe alarms. "normal" never sounds. */
  profiles: {
    warning: { freqs: [660], toneMs: 250, gapMs: 0, cycleMs: 3000, wave: "sine" },
    high: { freqs: [880, 880], toneMs: 180, gapMs: 120, cycleMs: 1500, wave: "square" },
    critical: { freqs: [1175, 880, 1175, 880], toneMs: 150, gapMs: 50, cycleMs: 1000, wave: "sawtooth" },
  } satisfies Record<Exclude<AlarmLevel, "normal">, SoundProfile>,
};
