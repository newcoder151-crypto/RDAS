import type { AlarmLevel } from "@/components/rdas/AlarmPanel";
import { ALARM_SOUND_CONFIG, type SoundProfile } from "./alarm-sound-config";

type ActiveTone = { osc: OscillatorNode; gain: GainNode };

/**
 * Synthesised alarm tones via the Web Audio API (no audio files needed).
 * Browsers keep an AudioContext suspended until the user interacts with the
 * page, so call unlock() from a click/keypress; start() works without one but
 * stays silent until the context is running.
 */
export class AlarmSound {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private level: AlarmLevel | null = null;
  private tones = new Set<ActiveTone>();

  /** Called whenever the AudioContext state changes (e.g. suspended -> running). */
  onStateChange: ((running: boolean) => void) | null = null;

  private ensureContext(): AudioContext | null {
    if (this.ctx) return this.ctx;
    if (typeof window === "undefined") return null;
    const Ctor: typeof AudioContext | undefined =
      window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    const ctx = new Ctor();
    const master = ctx.createGain();
    master.gain.value = ALARM_SOUND_CONFIG.volume;
    master.connect(ctx.destination);
    ctx.onstatechange = () => this.onStateChange?.(ctx.state === "running");
    this.ctx = ctx;
    this.master = master;
    this.onStateChange?.(ctx.state === "running");
    return ctx;
  }

  /** Resume the AudioContext. Must be called from a user gesture the first time. */
  async unlock(): Promise<boolean> {
    const ctx = this.ensureContext();
    if (!ctx) return false;
    if (ctx.state !== "running") {
      try {
        await ctx.resume();
      } catch {
        /* still blocked */
      }
    }
    return ctx.state === "running";
  }

  /** Start (or switch) the repeating alarm pattern for a severity level. */
  start(level: AlarmLevel): void {
    if (level === "normal") {
      this.stop();
      return;
    }
    if (this.level === level && this.timer) return;
    this.stop();
    const ctx = this.ensureContext();
    if (!ctx) return;
    this.level = level;
    ctx.resume().catch(() => {});
    const profile = ALARM_SOUND_CONFIG.profiles[level];
    const tick = () => this.playCycle(profile);
    tick();
    this.timer = setInterval(tick, profile.cycleMs);
  }

  /** Stop immediately, including any tone that is mid-play. */
  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.level = null;
    const ctx = this.ctx;
    if (!ctx) return;
    const now = ctx.currentTime;
    for (const { osc, gain } of this.tones) {
      try {
        gain.gain.cancelScheduledValues(now);
        gain.gain.setValueAtTime(0, now);
        osc.stop(now + 0.01);
      } catch {
        /* already stopped */
      }
    }
    this.tones.clear();
  }

  dispose(): void {
    this.stop();
    this.onStateChange = null;
    this.ctx?.close().catch(() => {});
    this.ctx = null;
    this.master = null;
  }

  private playCycle(p: SoundProfile): void {
    const ctx = this.ctx;
    const master = this.master;
    // While the context is suspended, don't queue tones: they would all burst
    // out at once the moment the user finally unlocks audio.
    if (!ctx || !master || ctx.state !== "running") return;
    const toneSec = p.toneMs / 1000;
    const stepSec = (p.toneMs + p.gapMs) / 1000;
    const t0 = ctx.currentTime;
    p.freqs.forEach((freq, i) => {
      const start = t0 + i * stepSec;
      const end = start + toneSec;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = p.wave;
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(1, start + 0.01);
      gain.gain.setValueAtTime(1, Math.max(start + 0.01, end - 0.02));
      gain.gain.linearRampToValueAtTime(0, end);
      osc.connect(gain).connect(master);
      const tone = { osc, gain };
      this.tones.add(tone);
      osc.onended = () => {
        this.tones.delete(tone);
        osc.disconnect();
        gain.disconnect();
      };
      osc.start(start);
      osc.stop(end + 0.02);
    });
  }
}
