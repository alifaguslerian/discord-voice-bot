"""
Discord Voice Bot — persistent voice channel occupant.
Streams silent audio to prevent idle-kick, reconnects on any disconnect.
"""

import asyncio
import logging
import os
import random
import signal
import subprocess
import sys
from enum import Enum, auto

import discord
from discord.ext import tasks
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
TOKEN            = os.getenv("DISCORD_TOKEN")
VOICE_CHANNEL_ID = int(os.getenv("VOICE_CHANNEL_ID", "0"))

RECONNECT_BASE_DELAY  = 5    # seconds
RECONNECT_MAX_DELAY   = 300  # 5 minutes cap
RECONNECT_JITTER      = 3    # ±3s jitter to avoid thundering herd
MONITOR_INTERVAL      = 30   # how often background task checks voice state

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("voice-bot")


# ── Silent Audio Source ──────────────────────────────────────────────────────
class SilentAudioSource(discord.AudioSource):
    """
    Generates silent PCM frames via FFmpeg anullsrc.
    More reliable than read()ing /dev/zero directly — FFmpeg handles
    timing and format conversion properly.
    """

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._start_ffmpeg()

    def _start_ffmpeg(self):
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-ac", "2",
            "-ar", "48000",
            "-f", "s16le",
            "pipe:1",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def read(self) -> bytes:
        if self._process is None or self._process.poll() is not None:
            # FFmpeg died — restart it
            log.warning("FFmpeg process died, restarting...")
            self._start_ffmpeg()

        data = self._process.stdout.read(3840)  # 20ms frame @ 48kHz stereo s16le
        if not data:
            return b"\x00" * 3840  # fallback: pure silence bytes
        return data

    def is_opus(self) -> bool:
        return False  # raw PCM, discord.py will encode to Opus

    def cleanup(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None


# ── Bot ──────────────────────────────────────────────────────────────────────
class VoiceBot(discord.Client):

    def __init__(self):
        intents = discord.Intents.default()
        intents.voice_states = True
        super().__init__(intents=intents)

        self._target_channel_id  = VOICE_CHANNEL_ID
        self._voice_client: discord.VoiceClient | None = None
        self._audio_source: SilentAudioSource | None = None
        self._reconnect_delay    = RECONNECT_BASE_DELAY
        self._reconnecting       = False  # guard against concurrent reconnects

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def on_ready(self):
        log.info(f"Logged in as {self.user} (ID: {self.user.id})")
        log.info(f"Target voice channel ID: {self._target_channel_id}")
        await self._join_voice()
        self._monitor_loop.start()

    async def on_disconnect(self):
        log.warning("Gateway disconnected")

    async def on_resumed(self):
        log.info("Gateway resumed")

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ):
        # Only care about our own voice state
        if member.id != self.user.id:
            return

        was_connected = before.channel is not None
        is_connected  = after.channel is not None

        if was_connected and not is_connected:
            log.warning(f"Forcibly disconnected from voice channel. Reconnecting...")
            await self._schedule_reconnect()

        elif is_connected and after.channel.id != self._target_channel_id:
            log.warning(f"Moved to wrong channel ({after.channel.id}), moving back...")
            await self._schedule_reconnect()

    # ── Voice Management ─────────────────────────────────────────────────────

    async def _join_voice(self):
        """
        Attempt to join the target voice channel.
        Returns True on success, False on failure.
        """
        channel = self.get_channel(self._target_channel_id)

        if channel is None:
            log.error(
                f"Channel {self._target_channel_id} not found. "
                "Check VOICE_CHANNEL_ID and that the bot is in the server."
            )
            return False

        if not isinstance(channel, discord.VoiceChannel):
            log.error(f"Channel {self._target_channel_id} is not a voice channel.")
            return False

        # Clean up stale voice client if exists
        if self._voice_client is not None:
            await self._cleanup_voice()

        try:
            log.info(f"Joining voice channel: {channel.name} ({channel.id})")
            self._voice_client = await channel.connect(
                timeout=30,
                reconnect=False,  # we handle reconnect ourselves
                self_deaf=True,   # we don't need to hear anything
            )
            self._start_audio()
            self._reconnect_delay = RECONNECT_BASE_DELAY  # reset backoff on success
            log.info("Successfully joined and streaming silent audio.")
            return True

        except discord.ClientException as e:
            log.error(f"ClientException joining voice: {e}")
        except discord.opus.OpusNotLoaded:
            log.critical(
                "Opus library not loaded. Install libopus: apt install libopus0"
            )
        except asyncio.TimeoutError:
            log.error("Timed out connecting to voice channel.")
        except Exception as e:
            log.exception(f"Unexpected error joining voice: {e}")

        return False

    def _start_audio(self):
        """Start streaming silent audio. Safe to call if already playing."""
        if self._voice_client is None:
            return

        if self._voice_client.is_playing():
            return

        if self._audio_source is not None:
            self._audio_source.cleanup()

        self._audio_source = SilentAudioSource()
        self._voice_client.play(
            self._audio_source,
            after=self._on_audio_end,
        )
        log.info("Silent audio stream started.")

    def _on_audio_end(self, error: Exception | None):
        """Called by discord.py when audio playback ends (error or natural)."""
        if error:
            log.warning(f"Audio stream ended with error: {error}")
        else:
            log.info("Audio stream ended. Restarting...")

        # Restart audio — schedule on event loop since this is called from audio thread
        asyncio.run_coroutine_threadsafe(self._restart_audio(), self.loop)

    async def _restart_audio(self):
        """Restart audio if voice client is still alive."""
        if self._voice_client and self._voice_client.is_connected():
            await asyncio.sleep(0.5)  # brief pause before restart
            self._start_audio()

    async def _cleanup_voice(self):
        """Cleanly teardown audio and voice client."""
        if self._audio_source:
            self._audio_source.cleanup()
            self._audio_source = None

        if self._voice_client:
            try:
                if self._voice_client.is_connected():
                    await self._voice_client.disconnect(force=True)
            except Exception as e:
                log.debug(f"Error during voice cleanup: {e}")
            self._voice_client = None

    # ── Reconnect Logic ──────────────────────────────────────────────────────

    async def _schedule_reconnect(self):
        """Guard against concurrent reconnect attempts."""
        if self._reconnecting:
            log.debug("Reconnect already in progress, skipping.")
            return
        self._reconnecting = True
        try:
            await self._reconnect_with_backoff()
        finally:
            self._reconnecting = False

    async def _reconnect_with_backoff(self):
        """
        Exponential backoff reconnect loop.
        Keeps trying until success or bot shuts down.
        """
        attempt = 0
        while not self.is_closed():
            attempt += 1
            jitter  = random.uniform(-RECONNECT_JITTER, RECONNECT_JITTER)
            delay   = min(self._reconnect_delay + jitter, RECONNECT_MAX_DELAY)

            log.info(f"Reconnect attempt #{attempt} in {delay:.1f}s...")
            await asyncio.sleep(max(0, delay))

            if self.is_closed():
                break

            success = await self._join_voice()
            if success:
                log.info(f"Reconnected successfully on attempt #{attempt}.")
                return

            # Exponential backoff: 5 → 10 → 20 → 40 → ... → 300
            self._reconnect_delay = min(
                self._reconnect_delay * 2,
                RECONNECT_MAX_DELAY,
            )

        log.error("Bot closed during reconnect loop.")

    # ── Background Monitor ───────────────────────────────────────────────────

    @tasks.loop(seconds=MONITOR_INTERVAL)
    async def _monitor_loop(self):
        """
        Heartbeat check — catches dropped connections that don't fire events.
        Discord occasionally drops voice without triggering on_voice_state_update.
        """
        vc = self._voice_client

        if vc is None or not vc.is_connected():
            log.warning("Monitor: voice client not connected. Triggering reconnect.")
            await self._schedule_reconnect()
            return

        if not vc.is_playing() and not vc.is_paused():
            log.warning("Monitor: audio stopped unexpectedly. Restarting audio.")
            self._start_audio()

    @_monitor_loop.before_loop
    async def _before_monitor(self):
        await self.wait_until_ready()

    # ── Shutdown ─────────────────────────────────────────────────────────────

    async def close(self):
        log.info("Shutting down...")
        self._monitor_loop.cancel()
        await self._cleanup_voice()
        await super().close()


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    if not TOKEN:
        log.critical("DISCORD_TOKEN not set in .env")
        sys.exit(1)

    if not VOICE_CHANNEL_ID:
        log.critical("VOICE_CHANNEL_ID not set or invalid in .env")
        sys.exit(1)

    bot = VoiceBot()

    # Graceful shutdown on SIGTERM/SIGINT (systemd sends SIGTERM)
    loop = asyncio.get_event_loop()

    def _shutdown(sig):
        log.info(f"Received {sig.name}, shutting down gracefully...")
        loop.create_task(bot.close())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown, sig)

    try:
        loop.run_until_complete(bot.start(TOKEN))
    except discord.LoginFailure:
        log.critical("Invalid Discord token. Check DISCORD_TOKEN in .env")
        sys.exit(1)
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        loop.close()


if __name__ == "__main__":
    main()