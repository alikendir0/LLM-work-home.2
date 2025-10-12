import os
import time
from google import genai
from google.genai import types, errors

# === CONFIGURATION ===
MODEL = "gemini-2.5-flash"          # You can change this if you have access to pro
MAX_TOKENS = 128_000                # Safe default context window
WARN_THRESHOLD = 0.8                # 80% usage triggers warning
MAX_RETRIES = 3                     # Retry attempts for rate limits

# === GLOBAL TOKEN COUNTERS ===
PROMPT_TOKENS_TOTAL = 0
RESPONSE_TOKENS_TOTAL = 0


# === UTILS ===
def warn(msg: str):
    print(f"⚠️  {msg}")

def info(msg: str):
    print(f"[INFO] {msg}")

def ok(msg: str):
    print(f"[OK] {msg}")


# === INITIALIZE CLIENT ===
def init_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment variables.")
    client = genai.Client(api_key=api_key)
    ok(f"Gemini client initialized with model '{MODEL}'")
    return client


# === TOKEN COUNT HELPERS ===
def count_tokens(client, contents) -> int:
    """Return token count for given contents, or 0 on failure."""
    try:
        resp = client.models.count_tokens(model=MODEL, contents=contents)
        return int(getattr(resp, "total_tokens", 0))
    except Exception as e:
        warn(f"Token counting failed: {e}. Proceeding without token count.")
        return 0

def check_context_window(client, contents_for_count):
    """Count tokens for a would-be request (e.g., prompt or chat history) and warn if near limit."""
    total = count_tokens(client, contents_for_count)
    if total:
        info(f"Current token usage estimate: {total}/{MAX_TOKENS}")
        if total >= MAX_TOKENS * WARN_THRESHOLD:
            warn(f"Context window nearly full ({total}/{MAX_TOKENS}).")
    return total


# === RATE LIMIT HANDLER ===
def safe_request(func):
    """Decorator for exponential backoff on code 429 (RESOURCE_EXHAUSTED)."""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except errors.APIError as e:
                if getattr(e, "code", None) == 429:
                    delay = 2 ** attempt
                    warn(f"Rate limit hit. Retrying in {delay}s (attempt {attempt+1})...")
                    time.sleep(delay)
                else:
                    raise
        raise RuntimeError("Max retry limit reached. Exiting.")
    return wrapper


# === TEXT GENERATION ===
@safe_request
def generate_text(client, prompt: str):
    """Generate text, track tokens (prompt & response), and monitor context usage."""
    global PROMPT_TOKENS_TOTAL, RESPONSE_TOKENS_TOTAL

    # Prompt tokens (only the prompt itself so we can report distinct totals)
    prompt_tokens = count_tokens(client, prompt)
    PROMPT_TOKENS_TOTAL += prompt_tokens
    info(f"[TEXT] Prompt tokens: {prompt_tokens}")

    # Context warning (optionally consider only prompt here; you can switch to a
    # richer context if you decide to prepend system/instructions later)
    check_context_window(client, prompt)

    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(),  # can set max_output_tokens if desired
    )
    text = resp.text or ""

    # Response tokens
    response_tokens = count_tokens(client, text) if text else 0
    RESPONSE_TOKENS_TOTAL += response_tokens
    info(f"[TEXT] Response tokens: {response_tokens}")

    return text


# === CHAT MODE ===
class ChatSession:
    def __init__(self, client):
        self.client = client
        self.chat = client.chats.create(model=MODEL)
        self.history = []  # local mirror for counting only

    @safe_request
    def send(self, message: str):
        """Send a chat message, track prompt/response tokens distinctly, and warn on context."""
        global PROMPT_TOKENS_TOTAL, RESPONSE_TOKENS_TOTAL

        # Count tokens for this *user message* (distinct prompt accounting)
        prompt_tokens = count_tokens(self.client, message)
        PROMPT_TOKENS_TOTAL += prompt_tokens
        info(f"[CHAT] Prompt tokens (this user msg): {prompt_tokens}")

        # For context warning, consider accumulated transcript + new user msg
        full_context = "\n".join(self.history + [f"User: {message}"])
        check_context_window(self.client, full_context)

        # Send message
        resp = self.chat.send_message(message)
        text = resp.text or ""

        # Count tokens for this *model response* (distinct response accounting)
        response_tokens = count_tokens(self.client, text) if text else 0
        RESPONSE_TOKENS_TOTAL += response_tokens
        info(f"[CHAT] Response tokens (this model msg): {response_tokens}")

        # Update local history mirror
        self.history.extend([f"User: {message}", f"Model: {text}"])
        return text


# === MAIN TESTING LOGIC ===
def main():
    client = init_client()

    print("\n--- TEXT GENERATION TEST ---")
    prompt = "Explain the difference between rate limits and context windows in simple terms."
    output = generate_text(client, prompt)
    print("\nModel Output:\n", output)

    print("\n--- CHAT MODE TEST ---")
    chat = ChatSession(client)
    r1 = chat.send("Hello! Tell me a short story about testing APIs.")
    print("\nChat 1:", r1)
    r2 = chat.send("Summarize that story in one sentence and mention 'context window'.")
    print("\nChat 2:", r2)

    # === FINAL TOKEN SUMMARY ===
    print("\n" + "-" * 64)
    ok("FINAL TOKEN SUMMARY")
    print(f"Total PROMPT tokens   : {PROMPT_TOKENS_TOTAL}")
    print(f"Total RESPONSE tokens : {RESPONSE_TOKENS_TOTAL}")
    print("-" * 64)


if __name__ == "__main__":
    main()
