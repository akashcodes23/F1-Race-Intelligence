"""
F1 Race Intelligence — Broadcast Audio & Pit-Wall Radio Chime Engine

Generates synthesized Web Audio telemetry cues (Box-Box-Box radio chime,
Safety Car caution horns, undercut alert bells) with zero external MP3 dependencies.
"""

import streamlit as st


def render_audio_controller():
    """
    Renders an audio preferences toggle in the Streamlit sidebar.
    """
    if "audio_enabled" not in st.session_state:
        st.session_state["audio_enabled"] = True

    audio_on = st.sidebar.toggle(
        "🔊 Pit-Wall Audio Effects",
        value=st.session_state["audio_enabled"],
        help="Enables synthesized F1 team radio pit chimes ('Box, Box') and Safety Car alerts.",
    )
    st.session_state["audio_enabled"] = audio_on
    return audio_on


def play_pit_radio_chime():
    """
    Synthesizes the authentic Formula 1 telemetry radio pit chime:
    Two-tone harmonic beep (880 Hz -> 1760 Hz) followed by radio acoustic burst.
    """
    if not st.session_state.get("audio_enabled", True):
        return

    html_code = """
    <script>
    (function() {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const now = ctx.currentTime;

            // Tone 1: 880 Hz (A5)
            const osc1 = ctx.createOscillator();
            const gain1 = ctx.createGain();
            osc1.type = 'sine';
            osc1.frequency.setValueAtTime(880, now);
            gain1.gain.setValueAtTime(0.18, now);
            gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.12);
            osc1.connect(gain1);
            gain1.connect(ctx.destination);
            osc1.start(now);
            osc1.stop(now + 0.12);

            // Tone 2: 1760 Hz (A6) Pit alert
            const osc2 = ctx.createOscillator();
            const gain2 = ctx.createGain();
            osc2.type = 'sine';
            osc2.frequency.setValueAtTime(1760, now + 0.14);
            gain2.gain.setValueAtTime(0.22, now + 0.14);
            gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
            osc2.connect(gain2);
            gain2.connect(ctx.destination);
            osc2.start(now + 0.14);
            osc2.stop(now + 0.35);
        } catch(e) {
            console.log("AudioContext blocked or unavailable", e);
        }
    })();
    </script>
    """
    st.components.v1.html(html_code, height=0, width=0)


def play_caution_warning():
    """
    Synthesizes Safety Car / VSC double cautionary horn pulse (440 Hz / 330 Hz).
    """
    if not st.session_state.get("audio_enabled", True):
        return

    html_code = """
    <script>
    (function() {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const now = ctx.currentTime;

            // Pulse 1: 440 Hz
            const osc1 = ctx.createOscillator();
            const gain1 = ctx.createGain();
            osc1.type = 'triangle';
            osc1.frequency.setValueAtTime(440, now);
            gain1.gain.setValueAtTime(0.25, now);
            gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
            osc1.connect(gain1);
            gain1.connect(ctx.destination);
            osc1.start(now);
            osc1.stop(now + 0.22);

            // Pulse 2: 330 Hz
            const osc2 = ctx.createOscillator();
            const gain2 = ctx.createGain();
            osc2.type = 'triangle';
            osc2.frequency.setValueAtTime(330, now + 0.25);
            gain2.gain.setValueAtTime(0.25, now + 0.25);
            gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.50);
            osc2.connect(gain2);
            gain2.connect(ctx.destination);
            osc2.start(now + 0.25);
            osc2.stop(now + 0.50);
        } catch(e) {
            console.log("AudioContext blocked or unavailable", e);
        }
    })();
    </script>
    """
    st.components.v1.html(html_code, height=0, width=0)
