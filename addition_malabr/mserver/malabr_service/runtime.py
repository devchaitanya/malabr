"""Server bootstrap: UDS listener, request routing, control reader thread.

Implements section 10a's three server-side rules:
  1. cross-origin eviction when a session is created
  2. every request ends in exactly ONE terminal frame, cancellation included
  3. reconcile on the control handshake, not only on explicit messages
"""

import os
import signal
import socket
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from .config import load_config
from .protocol import (
    FRAME_COMPLETE, FRAME_ERROR, FRAME_TOKEN,
    MSG_EXT_UNLOADED, MSG_FOREGROUND, MSG_LIVE_TABS, MSG_TAB_CLOSED,
    ProtocolError, ROUTE_CONTROL, ROUTE_GENERATE,
    encode_frame, read_control_message, recv_full, unpack_client_envelope,
)

# Section 7: this pool is a WAITING ROOM, not compute. It must sit well above
# n_seq_max or it becomes an invisible FIFO gate in front of the scheduler --
# a 9th concurrent tab would queue here, unparsed and invisible, even if it is
# the visible tab. One line, but load-bearing.
CLIENT_POOL_WORKERS = 64

# How long a handler waits for one frame before checking whether its session
# died underneath it. Bounds the "session torn down by another thread" case;
# the terminal frame normally arrives long before this.
OUTBOX_POLL_SECONDS = 0.25


class SingleInstanceError(RuntimeError):
    pass


class PidLock:
    """Refuse to start if another live instance holds the socket.

    The inherited code deleted a stale socket file before binding. That is safe
    ONLY if the previous process is genuinely dead. If a second app.py is ever
    launched while the first still lives -- a retry bug, or a browser crash and
    restart that does not guarantee the child died -- the newcomer would delete
    the LIVE instance's socket and steal the path, orphaning a process that
    still holds sessions and slots nobody can reach. Silently.

    So: check a pid file first, and only clear the socket once the owner is
    confirmed dead.
    """

    def __init__(self, path):
        self.path = path
        self._acquired = False

    def acquire(self):
        existing = self._read()
        if existing is not None and self._alive(existing):
            raise SingleInstanceError(
                f"another malabr server is running (pid {existing})")
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            fh.write(str(os.getpid()))
        os.replace(tmp, self.path)          # atomic
        self._acquired = True

    def release(self):
        if not self._acquired:
            return
        try:
            if self._read() == os.getpid():
                os.unlink(self.path)
        except OSError:
            pass
        self._acquired = False

    def _read(self):
        try:
            with open(self.path) as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _alive(pid):
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)                 # signal 0 tests existence only
        except ProcessLookupError:
            return False
        except PermissionError:
            return True                     # exists, owned by someone else
        return True


class ControlReader:
    """Owns the ONE control connection. Section 5e / 10a.

    A dedicated thread, separate from the request pool, permanently blocked on
    recv(). It exists because per-request sockets cannot carry these: an IDLE
    session has no open socket at all, so tab-close-while-idle and every
    visibility change had no channel to travel on.

    Write-only from the browser -- Python never replies.
    """

    def __init__(self, engine):
        self._engine = engine
        self._thread = None
        self._conn = None

    def serve(self, conn):
        """Take ownership of a control connection and read it to EOF.

        Only ONE is expected. A second replaces the first, because a browser
        that reconnected believes its new connection is authoritative and the
        old one is by definition stale.
        """
        old, self._conn = self._conn, conn
        if old is not None:
            try:
                old.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                old.close()
            except OSError:
                pass

        self._engine.set_control_connected(True)
        try:
            while True:
                try:
                    msg = read_control_message(conn)
                except ProtocolError as exc:
                    # A malformed control message is not recoverable: we cannot
                    # know where the next message starts in the stream. Drop the
                    # connection and let the browser's jittered reconnect (and
                    # its LIVE_TABS resync) restore correct state.
                    print(f"malabr: bad control message, dropping: {exc}",
                          file=sys.stderr)
                    return
                if msg is None:
                    return                  # browser closed
                self.dispatch(msg)
        finally:
            if self._conn is conn:
                self._conn = None
                # §5f: start the staleness clock. The browser reconnects with
                # jitter and resyncs via LIVE_TABS + FOREGROUND, so this is a
                # window, not a permanent state.
                self._engine.set_control_connected(False)
            try:
                conn.close()
            except OSError:
                pass

    def dispatch(self, msg):
        """Apply one control message. Unknown tab/extension is a NO-OP.

        Section 10a is explicit that this is defined behaviour, not an error:
        the browser does not track which sessions exist, so it will routinely
        name tabs we have never seen.
        """
        eng = self._engine
        if msg.kind == MSG_FOREGROUND:
            # A single attribute write. Atomic under the GIL, no lock needed,
            # and read fresh by build_batch on the very next round (§5f).
            eng.foreground_tab_id = msg.tab_id
            return

        if msg.kind == MSG_TAB_CLOSED:
            for key in eng.keys_for_tab(msg.tab_id):
                eng.cancel(key, "tab closed")
            return

        if msg.kind == MSG_LIVE_TABS:
            # RECONCILE, not replay. A TAB_CLOSED lost while the control
            # connection was down can NEVER be replayed -- once the tab is gone
            # the browser has no record it existed. So the browser pushes STATE
            # and we reap anything not in it.
            #
            # tab_id -1 means "not tab-scoped" and can never appear in a live
            # tab list, so it must be exempt or it would be reaped instantly.
            live = set(msg.tab_ids)
            for key in eng.all_keys():
                tab = key[1]
                if tab != -1 and tab not in live:
                    eng.cancel(key, "tab no longer live")
            return

        if msg.kind == MSG_EXT_UNLOADED:
            for key in eng.keys_for_extension(msg.extension_id):
                eng.cancel(key, "extension unloaded")
            return

    def start(self, conn):
        self._thread = threading.Thread(
            target=self.serve, args=(conn,), name="malabr-control", daemon=True)
        self._thread.start()


class MalabrServer:
    """UDS listener + request routing."""

    def __init__(self, cfg, engine, formatter_factory, output_cap):
        self._cfg = cfg
        self._engine = engine
        self._formatter_factory = formatter_factory
        self._output_cap = output_cap
        self._control = ControlReader(engine)
        self._lock = PidLock(cfg.socket_path + ".pid")
        self._server = None
        self._pool = None
        self._running = False

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        self._lock.acquire()                # raises if another instance lives
        # Only now is it safe to clear the socket: the pid lock has confirmed
        # no live owner. Doing this first, as the inherited code did, is what
        # made stealing a live instance's path possible.
        if os.path.exists(self._cfg.socket_path):
            os.unlink(self._cfg.socket_path)

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self._cfg.socket_path)
        self._server.listen(128)
        os.chmod(self._cfg.socket_path, 0o600)   # this user only
        self._pool = ThreadPoolExecutor(max_workers=CLIENT_POOL_WORKERS)
        self._running = True

    def stop(self):
        self._running = False
        for closer in (self._server,):
            if closer is not None:
                try:
                    closer.close()
                except OSError:
                    pass
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        if os.path.exists(self._cfg.socket_path):
            try:
                os.unlink(self._cfg.socket_path)
            except OSError:
                pass
        self._lock.release()

    def serve_forever(self):
        while self._running:
            try:
                conn, _ = self._server.accept()
            except OSError:
                if self._running:
                    raise
                return
            self._pool.submit(self._guard, conn)

    def _guard(self, conn):
        try:
            self.handle_connection(conn)
        except Exception as exc:                     # never kill a pool worker
            print(f"malabr: connection error: {exc}", file=sys.stderr)
            try:
                conn.close()
            except OSError:
                pass

    # -- routing -----------------------------------------------------------

    def handle_connection(self, conn):
        head = recv_full(conn, 4)
        if head is None:
            conn.close()
            return
        (header_len,) = struct.unpack(">I", head)
        header = recv_full(conn, header_len)
        if header is None:
            conn.close()
            return
        env = unpack_client_envelope(header)         # validates and bounds

        if env.route == ROUTE_CONTROL:
            # Hand the socket to the dedicated reader thread and do NOT close
            # it here -- it lives for the whole browser session.
            self._control.start(conn)
            return

        try:
            self.handle_generate(conn, env)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def handle_generate(self, conn, env):
        prompt = recv_full(conn, env.payload_size)   # size already bounded
        if prompt is None:
            return
        try:
            text = prompt.decode("utf-8")
        except UnicodeDecodeError:
            self._send(conn, FRAME_ERROR, "prompt is not valid UTF-8")
            return

        eng = self._engine
        # RULE 1: cross-origin eviction. Creating a session for (ext, tab,
        # origin) tears down any session with the same (ext, tab) and a
        # DIFFERENT origin. This is what makes §5g work with no navigation
        # observer -- the next request from that tab simply carries a different
        # origin and the old session cannot survive it.
        doomed = eng.evict_other_origins(env.extension_id, env.tab_id, env.origin)
        # Wait for the eviction we just asked for to actually complete. Without
        # this, a cross-origin navigation is rejected for "no free slots" while
        # the slot it needs belongs to the session we just condemned -- and it
        # is exactly the full-capacity case where that hurts most.
        eng.wait_for_teardown(doomed)

        session, created = eng.get_or_create(
            env.session_key, self._formatter_factory, self._output_cap)
        if session is None:
            # §7's named limitation: admission is not visibility-aware, so a
            # foreground tab CAN be rejected while background tabs hold slots.
            # Surfacing it beats failing silently.
            self._send(conn, FRAME_ERROR, "no free session slots")
            return
        if created and env.tab_id != -1 and env.is_foreground_seed \
                and eng.foreground_tab_id == -1:
            # Visibility is a SEED for a NEW session only, and only while the
            # control connection has not told us otherwise. It must never
            # override live state: §6's ordering rule does not guarantee the
            # seed is newer than the control channel's view.
            eng.foreground_tab_id = env.tab_id

        outbox = eng.submit(env.session_key, text)
        if outbox is None:
            self._send(conn, FRAME_ERROR, "session unavailable")
            return
        self._stream(conn, session, outbox)

    def _stream(self, conn, session, outbox):
        """Pump frames until exactly ONE terminal frame. Rule 2.

        A handler whose session was cancelled must emit a terminal frame and
        close, not block waiting for tokens that will never come -- otherwise
        the browser sits until its 60s SO_RCVTIMEO instead of ending promptly.
        """
        import queue as _q
        while True:
            try:
                frame_type, payload = outbox.get(timeout=OUTBOX_POLL_SECONDS)
            except _q.Empty:
                # Nothing arrived. If the session died without managing to
                # enqueue its terminal frame (teardown from another thread,
                # engine crash), synthesise one rather than hang.
                if session.state == "dead":
                    self._send(conn, FRAME_ERROR, "session ended")
                    return
                continue
            if not self._send(conn, frame_type, payload):
                return                       # peer gone; engine sees the drop
            if frame_type in (FRAME_COMPLETE, FRAME_ERROR):
                return

    @staticmethod
    def _send(conn, frame_type, payload):
        try:
            conn.sendall(encode_frame(frame_type, payload))
            return True
        except OSError:
            return False


def run_server(config=None, engine=None, formatter_factory=None,
               output_cap=None):
    cfg = config or load_config()
    server = MalabrServer(cfg, engine, formatter_factory, output_cap)
    server.start()

    def shutdown(*_):
        server.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("malabr server listening on", cfg.socket_path)
    try:
        server.serve_forever()
    finally:
        server.stop()
