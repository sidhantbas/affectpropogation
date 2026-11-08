"""
Fixed study parameters. Per the mission spec (Appendix B):
"generation temperature: fixed across all missions — temperature changes g"
"embedder: one model, never swapped mid study — swapping changes the floor"

Do not change these values mid-study. If a value here must change, that is a
new study: re-run Mission 0 from scratch and say so explicitly.
"""

# --- Generator (pinned) ---
PROVIDER = "gemini"
GEMINI_MODEL = "gemini-2.5-flash"
GENERATION_TEMPERATURE = 0.9

# Claude backend, held in reserve for a second-model replication of the
# protocol. Not used by Mission 0's default run.
CLAUDE_MODEL = "claude-sonnet-5"

# --- Embedder (pinned) ---
EMBEDDER_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- Node definition (fixed by fiat, spec section "Node definition") ---
NODE_INSTRUCTION = (
    "Write the next beat that follows causally from this one. "
    "Respond with exactly one short narrative beat, two to three sentences, "
    "describing a state of affairs. Do not include any preamble, labels, "
    "quotation marks, or commentary -- output only the beat itself."
)

SEED_INSTRUCTION = (
    "Write a short narrative beat, two to three sentences, describing a "
    "state of affairs. This is the opening beat of a story -- invent any "
    "setting, characters, and situation you like. Respond with exactly one "
    "beat: two to three sentences, no preamble, no labels, no quotation "
    "marks, no commentary."
)

# --- Rate limiting ---
MAX_WORKERS = 6
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 90.0

# --- Mission 0 run sizes ---
SMOKE_TEST_CHAINS = 3
SMOKE_TEST_DEPTH = 5

FULL_RUN_CHAINS = 60
FULL_RUN_DEPTH = 20

DATA_DIR = "data"
