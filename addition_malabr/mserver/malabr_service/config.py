"""Runtime configuration -- see design_doc/phase1_design.md section 11.

Sizing is COMPUTED AT STARTUP from detected hardware, not hardcoded. The
inherited config hardcoded n_ctx=4096, which at n_seq_max=8 gives 512 tokens
per session -- roughly 6-8 short chat turns before compaction fires on EVERY
conversation. That tests "does compaction work at all" rather than "does the
policy correctly decide whose context to shrink under contention", which is
the actual research question.
"""

import os
from dataclasses import dataclass, field


# Measured on this project's test models: ~107 KB of KV per token of context.
# Used to turn an available-RAM figure into a context size.
BYTES_PER_TOKEN = 107 * 1024

# Fraction of USABLE RAM the KV pool may claim. The pool is a STOCK, not a
# flow: n_ctx is allocated once as a single fixed block at context creation and
# there is no runtime path by which it can grow. RSS was measured flat (~1139MB)
# whether 1 or 8 sessions were resident, which is why no cgroup memory ceiling
# is needed on top -- the allocation already cannot exceed itself.
#
# SECTION 11 SAYS 0.8 OF MemAvailable. That is measurably wrong here and the
# number is not a rounding difference: on this machine it yields 75,340 tokens
# = 7.7 GB of KV, against the 16,384 (1.7 GB) section 7 states outright -- 4.6x
# apart. Three reasons the formula overshoots:
#   1. MemAvailable counts reclaimable page cache as free. True for transient
#      allocations, false for a PERMANENT reservation. Here MemAvailable is
#      ~9 GB of which almost all is cache, and swap is already full.
#   2. It never subtracts the model weights, which are also resident.
#   3. It leaves nothing for the browser -- and protecting the browser from the
#      inference engine is the entire point of the design.
# So: a smaller fraction, an explicit browser reserve, and the model subtracted.
RAM_FRACTION_FOR_KV = 0.5

# Never hand the KV pool memory the browser will need. Chromium with a handful
# of tabs sits comfortably above this; it is a floor, not a generous estimate.
BROWSER_RESERVE_BYTES = 2 * 1024 ** 3

# Reserve 20% of CPU for the browser hosting the engine. 100% is unrealistic:
# it starves the browser and everything else on the machine.
QUOTA_PCT = 80

# Concurrent tabs. A REAL TRADEOFF, not a free parameter: more slots divides
# less context per tab, whatever n_ctx turns out to be.
DEFAULT_N_SEQ_MAX = 8

# Floor and ceiling on the computed context size. The floor keeps a small
# machine from computing a context so tight that compaction is permanent; the
# ceiling stops a large machine from reserving absurd amounts of RAM for a
# feature the user may never use.
MIN_N_CTX = 4096
# Section 7 states n_ctx = 16384 outright. Used as the default ceiling so the
# computed value agrees with the document on reference hardware instead of
# exceeding it by 4x; raise it deliberately via MALABR_N_CTX on a machine with
# real headroom.
MAX_N_CTX = 16384


def detect_physical_cores():
    """Physical, not logical. Measured on this project's machine: 8 logical /
    4 physical, and 8 threads is SLOWER than 4 -- hyperthread contention, not
    parallelism. Counting logical CPUs here would bake that loss in."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    # /proc/cpuinfo: count distinct (physical id, core id) pairs.
    try:
        pairs, phys, core = set(), None, None
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip() and phys is not None and core is not None:
                    pairs.add((phys, core)); phys = core = None
        if phys is not None and core is not None:
            pairs.add((phys, core))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    logical = os.cpu_count() or 1
    return max(1, logical // 2)          # assume SMT rather than overcommit


def detect_available_ram_bytes():
    """MemAvailable, not MemTotal -- the engine shares the machine."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 2 * 1024 ** 3                  # conservative fallback


def compute_n_ctx(n_seq_max=DEFAULT_N_SEQ_MAX, available_bytes=None,
                  model_bytes=0):
    avail = available_bytes if available_bytes is not None else detect_available_ram_bytes()
    usable = avail - BROWSER_RESERVE_BYTES - model_bytes
    if usable <= 0:
        return MIN_N_CTX                  # tiny machine: floor, do not go negative
    raw = int(usable * RAM_FRACTION_FOR_KV / BYTES_PER_TOKEN)
    n_ctx = max(MIN_N_CTX, min(raw, MAX_N_CTX))
    # llama.cpp divides the pool evenly, so keep n_ctx a whole multiple of
    # n_seq_max; otherwise the remainder is silently unusable.
    return (n_ctx // n_seq_max) * n_seq_max


def compute_n_threads(physical_cores=None):
    """Derived from the SAME quota as the cgroup limit, deliberately.

    Setting one without the other is the nonlinear-loss regime measured in
    section 11: squeezing more threads than the quota allows forces the kernel
    to preempt and context-switch them inside a shrunk slice -- pure overhead,
    no extra work done.
    """
    cores = physical_cores if physical_cores is not None else detect_physical_cores()
    return max(1, round(cores * QUOTA_PCT / 100))


@dataclass(frozen=True)
class ServerConfig:
    socket_path: str
    base_dir: str
    model_dir: str
    model_path: str
    calibration_path: str

    n_ctx: int
    n_seq_max: int
    n_threads: int
    n_batch: int
    quota_pct: int

    client_pool_workers: int

    # Shared-KV mode (MALABR_SHARED_KV=1). Off: the KV cache is pre-partitioned
    # into n_seq_max equal hard slices and a session cannot exceed n_ctx/n_seq_max
    # (llama.cpp enforces this). On: kv_unified=true, one shared pool of n_ctx
    # cells, each session gets a LARGER soft cap (n_ctx // shared_kv_soft_div,
    # default n_ctx/2) and compaction is driven by AGGREGATE pressure -- so 1-2
    # tabs get a big budget and many tabs degrade to roughly fair share.
    shared_kv: bool = False
    shared_kv_soft_div: int = 2

    @property
    def n_ctx_per_session(self):
        """The per-session budget section 8's compaction trigger measures against.

        Shared-KV: a SOFT cap (n_ctx/2 by default) -- the aggregate cap does the
        real bounding. Partitioned: the HARD slice llama.cpp enforces.
        """
        if self.shared_kv:
            return max(self.n_ctx // self.n_seq_max,
                       self.n_ctx // max(1, self.shared_kv_soft_div))
        return self.n_ctx // self.n_seq_max


def list_models(model_dir):
    """Every *.gguf in the model directory, by bare name (no extension).

    The chat panel's model switcher offers exactly this set; a switch names one
    of these and the server re-execs with MALABR_MODEL_PATH pointed at it.
    """
    try:
        names = sorted(f[:-5] for f in os.listdir(model_dir)
                       if f.endswith(".gguf"))
    except OSError:
        names = []
    return names


def resolve_model(model_dir, name):
    """Map a bare model name from list_models() back to a full path.

    Returns None if the name is not one of the directory's .gguf files -- the
    switcher must never be able to make the server exec an arbitrary path.
    """
    if name not in list_models(model_dir):
        return None
    return os.path.join(model_dir, name + ".gguf")


def load_config(base_dir=None):
    # root is the mserver/ package dir; models live one level up beside it, in
    # addition_malabr/models. Deriving model_dir from `root` put it in
    # mserver/models, where nothing is -- calibration fell straight through to
    # its Phase E fallback and reported a conservative curve as if measured.
    root = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n_seq_max = int(os.getenv("MALABR_N_SEQ_MAX", DEFAULT_N_SEQ_MAX))

    env_ctx = os.getenv("MALABR_N_CTX")
    model_dir = os.getenv("MALABR_MODEL_DIR",
                          os.path.join(os.path.dirname(root), "models"))
    model_path = os.getenv("MALABR_MODEL_PATH",
                           os.path.join(model_dir, "Qwen3-0.6B-Q8_0.gguf"))
    try:
        model_bytes = os.path.getsize(model_path)
    except OSError:
        model_bytes = 0
    n_ctx = ((int(env_ctx) // n_seq_max) * n_seq_max if env_ctx
             else compute_n_ctx(n_seq_max, model_bytes=model_bytes))

    return ServerConfig(
        socket_path=os.getenv("MALABR_SOCKET_PATH", "/tmp/malabr_v3.sck"),
        base_dir=root,
        model_dir=model_dir,
        model_path=model_path,
        # Section 4's ONE stated exception to "nothing durable": calibration
        # data is written to disk on purpose, so a cold start does not have to
        # re-measure the machine every time.
        calibration_path=os.getenv("MALABR_CALIBRATION_PATH",
                                   os.path.join(root, "calibration.json")),
        n_ctx=n_ctx,
        n_seq_max=n_seq_max,
        n_threads=int(os.getenv("MALABR_N_THREADS", compute_n_threads())),
        n_batch=int(os.getenv("MALABR_N_BATCH", "512")),
        quota_pct=int(os.getenv("MALABR_QUOTA_PCT", QUOTA_PCT)),
        # Section 7: a WAITING ROOM, not compute. Must sit well above n_seq_max
        # or the pool becomes an invisible FIFO gate in front of the scheduler.
        client_pool_workers=int(os.getenv("MALABR_CLIENT_WORKERS", "64")),
        shared_kv=os.getenv("MALABR_SHARED_KV") == "1",
        shared_kv_soft_div=int(os.getenv("MALABR_SESSION_SOFT_DIV", "2")),
    )
