import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Alarm, AlarmLevel } from "@/components/rdas/AlarmPanel";
import { AlarmSound } from "@/lib/alarm-sound";
import { ALARM_SOUND_CONFIG } from "@/lib/alarm-sound-config";

const rank: Record<AlarmLevel, number> = { normal: 0, warning: 1, high: 2, critical: 3 };

function readMuted(): boolean {
  try {
    return localStorage.getItem(ALARM_SOUND_CONFIG.muteStorageKey) === "1";
  } catch {
    return false;
  }
}

export type AlarmSoundState = {
  /** True while the siren is actually supposed to be sounding. */
  sounding: boolean;
  muted: boolean;
  /** Browser is blocking audio until the user clicks something. */
  blocked: boolean;
  /** Click handler for the sound button: unlocks audio if blocked, otherwise toggles mute. */
  toggle: () => void;
};

/**
 * Plays the alarm tone while at least one active alarm is NOT acknowledged,
 * using the severity of the worst unacknowledged alarm. It stops as soon as
 * every active alarm has been acknowledged (or cleared).
 *
 * @param enabled false (e.g. demo data / no backend) keeps it silent.
 */
export function useAlarmSound(alarms: Alarm[], enabled: boolean): AlarmSoundState {
  const soundRef = useRef<AlarmSound | null>(null);
  const [muted, setMuted] = useState<boolean>(readMuted);
  const [blocked, setBlocked] = useState(false);

  if (soundRef.current === null && typeof window !== "undefined") {
    soundRef.current = new AlarmSound();
  }

  const topUnacked = useMemo<AlarmLevel | null>(() => {
    let best: AlarmLevel | null = null;
    for (const a of alarms) {
      if (a.level === "normal" || a.acked) continue;
      if (best === null || rank[a.level] > rank[best]) best = a.level;
    }
    return best;
  }, [alarms]);

  const sounding = enabled && !muted && topUnacked !== null;

  useEffect(() => {
    const s = soundRef.current;
    if (!s) return;
    s.onStateChange = (running) => setBlocked(!running);
    // Browsers only allow audio after a user gesture: unlock on the first one.
    const unlock = () => {
      void s.unlock();
    };
    window.addEventListener("pointerdown", unlock);
    window.addEventListener("keydown", unlock);
    return () => {
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
      s.dispose();
    };
  }, []);

  useEffect(() => {
    const s = soundRef.current;
    if (!s) return;
    if (sounding && topUnacked) s.start(topUnacked);
    else s.stop();
  }, [sounding, topUnacked]);

  const toggle = useCallback(() => {
    const s = soundRef.current;
    if (!s) return;
    if (blocked) {
      void s.unlock();
      return;
    }
    setMuted((m) => {
      const next = !m;
      try {
        localStorage.setItem(ALARM_SOUND_CONFIG.muteStorageKey, next ? "1" : "0");
      } catch {
        /* storage unavailable: mute only lasts for this session */
      }
      return next;
    });
  }, [blocked]);

  return { sounding, muted, blocked, toggle };
}
