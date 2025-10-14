import os
import time
import asyncio
import inspect
from google import genai
from google.genai import types, errors
<<<<<<< HEAD
from google.api_core import exceptions

=======
#selam
>>>>>>> 0f9e08c66a29fdc0bae70c2ad14965a60c9ad8ec
# === CONFIGURATION ===
MODEL = "gemini-2.5-flash-lite"          # You can change this if you have access to pro
MAX_TOKENS = 250_000                # Safe default context window
WARN_THRESHOLD = 0.8                # 80% usage triggers warning
MAX_RETRIES = 3                     # Retry attempts for rate limits
RETRY_BASE_DELAY = 2
API_DELAY_SUBTRACTION = 20            # Seconds to subtract from API-suggested retry delay to optimize wait time

# === GLOBAL TOKEN COUNTERS ===
PROMPT_TOKENS_TOTAL = 0
RESPONSE_TOKENS_TOTAL = 0


# === UTILS ===
def warn(msg): print(f"\033[93m[WARNING]\033[0m {msg}")
def info(msg): print(f"\033[94m[INFO]\033[0m {msg}")
def ok(msg): print(f"\033[92m[OK]\033[0m {msg}")
def error(msg): print(f"\033[91m[ERROR]\033[0m {msg}")

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
        maybe = client.models.count_tokens(model=MODEL, contents=contents)
        
        # Handle async count_tokens
        if inspect.isawaitable(maybe):
            try:
                resp = asyncio.run(maybe)  # type: ignore
            except RuntimeError:
                warn("Token counting skipped because an event loop is already running.")
                return 0
        else:
            resp = maybe
        
        # Extract total_tokens from response
        total = getattr(resp, "total_tokens", None)
        if total is None and hasattr(resp, "get"):
            total = resp.get("total_tokens", 0)
        
        return int(total) if total else 0
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


# === ENHANCED RATE LIMIT HANDLER ===
def extract_retry_delay(exception) -> float:
    """Extract retry delay from API error details"""
    try:
        # Try to get from exception attributes
        if hasattr(exception, 'retry_after'):
            return float(exception.retry_after)
        
        # Try to parse from error message/details
        error_str = str(exception)
        
        # Look for "Please retry in X.Xs" or "retryDelay: Xs"
        import re
        
        # Pattern 1: "Please retry in 58.312898435s"
        match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # Pattern 2: "Please retry in 646.143327ms" (milliseconds)
        match = re.search(r'retry in (\d+(?:\.\d+)?)ms', error_str, re.IGNORECASE)
        if match:
            return float(match.group(1)) / 1000  # Convert ms to seconds
        
        # Pattern 3: "retryDelay': 'XXs'" or "'retryDelay': 'XX.XXs'"
        match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", error_str)
        if match:
            return float(match.group(1))
        
        # Pattern 4: Check if error has details attribute (for google.api_core exceptions)
        if hasattr(exception, 'details'):
            details_str = str(exception.details)
            match = re.search(r'retryDelay.*?(\d+)s', details_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
    
    except Exception:
        pass
    
    return 0


def safe_request(func):
    """Decorator to handle rate limits with exponential backoff using API-suggested delays"""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if this is a rate limit error
                error_str = str(e).lower()
                is_rate_limit = (
                    isinstance(e, (exceptions.ResourceExhausted, exceptions.TooManyRequests)) or
                    any(keyword in error_str for keyword in ["rate limit", "quota", "429"])
                )
                
                if not is_rate_limit:
                    error(f"Non-retryable error: {e}")
                    raise
                
                # Handle rate limit error
                warn(f"Rate limit detected: {type(e).__name__}")
                
                # Extract and calculate retry delay
                api_retry_delay = extract_retry_delay(e)
                
                if api_retry_delay > 0:
                    # Use API's suggested delay minus 20 seconds (minimum 1 second)
                    delay = max(1, api_retry_delay - API_DELAY_SUBTRACTION)
                    info(f"API suggests retry after: {api_retry_delay:.1f}s")
                    info(f"Using optimized delay: {delay:.1f}s (API delay - {API_DELAY_SUBTRACTION:.1f}ss)")
                else:
                    # Fallback to exponential backoff
                    delay = RETRY_BASE_DELAY ** (attempt + 1)
                    info(f"No API retry delay found, using exponential backoff: {delay}s")
                
                # Last attempt check
                if attempt == MAX_RETRIES - 1:
                    error(f"Max retry limit ({MAX_RETRIES}) reached.")
                    raise RuntimeError("Max retry limit reached after rate limit errors.")
                
                # Wait and retry
                info(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
        
        # This should never be reached due to the raise in the loop
        error(f"Max retry limit ({MAX_RETRIES}) reached.")
        raise RuntimeError("Max retry limit reached after rate limit errors.")
    
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


# === STRESS TEST FUNCTIONS ===
def measure_actual_rpm(client):
    """
    Measure the actual RPM (Requests Per Minute) by running for exactly 2 minutes.
    Ignores API retry delay suggestions and measures real throughput.
    """
    print("\n" + "=" * 70)
    print("ACTUAL RPM MEASUREMENT TEST - 2 MINUTE STRESS TEST")
    print("=" * 70)
    
    info("Running continuous requests for 2 minutes to measure actual RPM...")
    info("This test will use retry logic to maximize successful requests.")
    
    test_duration = 120  # 2 minutes in seconds
    start_time = time.time()
    end_time = start_time + test_duration
    
    successful_requests = 0
    failed_requests = 0
    total_attempts = 0
    
    request_log = []
    
    start_time_str = time.strftime('%H:%M:%S', time.localtime(start_time))
    info(f"Test started at: {start_time_str}")
    info(f"Test will run until: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    
    print("\n" + "-" * 70)
    
    while time.time() < end_time:
        total_attempts += 1
        current_time = time.time()
        elapsed = current_time - start_time
        remaining = end_time - current_time
        
        try:
            info(f"[{elapsed:.1f}s elapsed, {remaining:.1f}s remaining] Request #{total_attempts}")
            
            response = generate_text(
                client,
                f"Number {total_attempts}"
            )
            
            success_time = time.time()
            successful_requests += 1
            
            request_log.append({
                'attempt': total_attempts,
                'success': True,
                'time': success_time,
                'elapsed': success_time - start_time
            })
            
            ok(f"✓ Success #{successful_requests} at {time.strftime('%H:%M:%S', time.localtime(success_time))}")
            
        except Exception as e:
            failed_requests += 1
            error(f"✗ Request #{total_attempts} failed: {str(e)[:100]}")
            
            request_log.append({
                'attempt': total_attempts,
                'success': False,
                'time': time.time(),
                'elapsed': time.time() - start_time
            })
    
    actual_end_time = time.time()
    actual_duration = actual_end_time - start_time
    
    # Analysis
    print("\n" + "=" * 70)
    print("RPM MEASUREMENT RESULTS")
    print("=" * 70)
    
    print(f"\n⏱️  Test Duration:")
    print(f"   Start time:        {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print(f"   End time:          {time.strftime('%H:%M:%S', time.localtime(actual_end_time))}")
    print(f"   Actual duration:   {actual_duration:.2f} seconds ({actual_duration/60:.2f} minutes)")
    
    print(f"\n📊 Request Statistics:")
    print(f"   Total attempts:       {total_attempts}")
    print(f"   Successful requests:  {successful_requests}")
    print(f"   Failed requests:      {failed_requests}")
    print(f"   Success rate:         {(successful_requests/total_attempts*100):.1f}%")
    
    print(f"\n🎯 RPM Calculation:")
    rpm = (successful_requests / actual_duration) * 60
    print(f"   Requests per minute:  {rpm:.2f} RPM")
    print(f"   Requests per second:  {successful_requests/actual_duration:.2f} RPS")
    
    # Calculate requests per minute window
    print(f"\n📈 Per-Minute Breakdown:")
    for minute in range(int(actual_duration / 60) + 1):
        minute_start = start_time + (minute * 60)
        minute_end = minute_start + 60
        
        minute_requests = [log for log in request_log 
                          if log['success'] and minute_start <= start_time + log['elapsed'] < minute_end]
        
        print(f"   Minute {minute+1}: {len(minute_requests)} successful requests")
    
    print("\n" + "=" * 70)
    ok(f"MEASURED ACTUAL RPM: {rpm:.2f}")
    print("=" * 70)
    
    return rpm


def prove_rate_limit_window(client):
    """
    Prove whether Gemini's 15 RPM limit uses:
    - Rolling window (starts from first request)
    - Fixed window (aligned to clock time)
    
    Test strategy:
    1. Send 15 requests rapidly (should all succeed)
    2. Send 16th request immediately (should fail if 15 RPM enforced)
    3. Wait and retry to see when quota resets
    """
    print("\n" + "=" * 70)
    print("RATE LIMIT WINDOW DETECTION TEST")
    print("=" * 70)
    
    info("Testing Gemini's 15 requests/minute rate limit behavior...")
    info("Expected: 15 requests should succeed, 16th should fail")
    
    request_times = []
    success_count = 0
    first_failure_time = None
    
    # Phase 1: Send requests rapidly to hit the limit
    print("\n--- PHASE 1: Sending requests to hit 15 RPM limit ---")
    for i in range(100):  # Try 20 to definitely hit the limit
        request_start = time.time()
        request_time_str = time.strftime('%H:%M:%S', time.localtime(request_start))
        
        try:
            info(f"Request #{i+1} at {request_time_str}")
            response = generate_text(
                client,
                f"Say the number {i}."
            )
            
            # Capture actual success time (after any retries)
            success_time = time.time()
            success_time_str = time.strftime('%H:%M:%S', time.localtime(success_time))
            
            request_times.append({
                'num': i + 1,
                'time': success_time,
                'time_str': success_time_str,
                'success': True
            })
            success_count += 1
            ok(f"✓ Request #{i+1} succeeded at {success_time_str}")
            
        except Exception as e:
            request_times.append({
                'num': i + 1,
                'time': request_start,
                'time_str': request_time_str,
                'success': False,
                'error': str(e)
            })
            
            if first_failure_time is None:
                first_failure_time = request_start
                first_failure_request_num = i + 1
            
            error(f"✗ Request #{i+1} FAILED at {request_time_str}")
            warn(f"Error: {e}")
            
            # If we hit rate limit, stop sending more requests
            error_str = str(e).lower()
            if any(x in error_str for x in ["rate limit", "quota", "429", "resource exhausted"]):
                warn(f"Rate limit hit after {success_count} successful requests")
                break
    
    first_request_time = request_times[0]['time']
    first_request_str = request_times[0]['time_str']
    
    print(f"\n📊 Summary:")
    print(f"   First request:  #{1} at {first_request_str}")
    print(f"   Total successful: {success_count}")
    print(f"   Total failed: {len(request_times) - success_count}")

def stress_test_rate_limits(client):
    """Test rate limit handling with rapid requests"""
    print("\n" + "=" * 70)
    print("STRESS TEST: RATE LIMITS")
    print("=" * 70)
    
    info("Sending rapid API requests to trigger rate limits...")
    
    # Send many rapid requests
    num_requests = 30
    successful = 0
    rate_limited = 0
    
    first_request_time = None
    last_rate_error_time = None
    
    for i in range(num_requests):
        try:
            # Log first request time
            if first_request_time is None:
                first_request_time = time.time()
                info(f"First request started at: {time.strftime('%H:%M:%S', time.localtime(first_request_time))}")
            
            info(f"\nRequest {i+1}/{num_requests}")
            response = generate_text(
                client,
                f"Give me one interesting fact about the number {i}. Be brief."
            )
            print(f"Response: {response[:80]}...")
            successful += 1
            time.sleep(0.1)  # Small delay
        except (exceptions.ResourceExhausted, exceptions.TooManyRequests) as e:
            # Log rate limit error time
            last_rate_error_time = time.time()
            warn(f"Request {i+1} hit rate limit at: {time.strftime('%H:%M:%S', time.localtime(last_rate_error_time))}")
            warn(f"Request {i+1} failed permanently: {e}")
            rate_limited += 1
        except Exception as e:
            # Check if it's a rate limit error in disguise
            error_str = str(e).lower()
            if any(x in error_str for x in ["rate limit", "quota", "429"]):
                last_rate_error_time = time.time()
                warn(f"Request {i+1} hit rate limit at: {time.strftime('%H:%M:%S', time.localtime(last_rate_error_time))}")
            warn(f"Request {i+1} failed permanently: {e}")
            rate_limited += 1
    
    print("\n" + "-" * 70)
    ok(f"Rate limit test complete: {successful} successful, {rate_limited} failed")
    
    # Summary of timing
    if first_request_time:
        info(f"First request time: {time.strftime('%H:%M:%S', time.localtime(first_request_time))}")
    if last_rate_error_time:
        info(f"Last rate error time: {time.strftime('%H:%M:%S', time.localtime(last_rate_error_time))}")
        if first_request_time:
            elapsed = last_rate_error_time - first_request_time
            info(f"Time between first request and last rate error: {elapsed:.2f}s")



def stress_test_context_window(client):
    """Test context window warning system"""
    print("\n" + "=" * 70)
    print("STRESS TEST: CONTEXT WINDOW")
    print("=" * 70)
    
    info("Building large prompt to test context window limits...")
    
    # Create progressively larger prompts
    base_text = "Artificial intelligence is transforming the world. " * 100
    
    for multiplier in [1, 10, 50, 100]:
        large_prompt = base_text * multiplier
        token_count = count_tokens(client, large_prompt)
        
        info(f"\nTest with {token_count:,} tokens (multiplier: {multiplier})")
        
        if token_count < MAX_TOKENS * 0.9:
            try:
                response = generate_text(
                    client,
                    f"Summarize this in one sentence: {large_prompt[:1000]}"
                )
                print(f"Response: {response[:100]}...")
            except Exception as e:
                warn(f"Generation failed: {e}")
        else:
            warn("Skipping generation - would exceed context window")
    
    ok("Context window test complete")


def stress_test_chat_history(client):
    """Test chat with long conversation history"""
    print("\n" + "=" * 70)
    print("STRESS TEST: CHAT HISTORY")
    print("=" * 70)
    
    chat = ChatSession(client)
    
    topics = [
        "What is machine learning?",
        "Explain deep learning briefly.",
        "What are neural networks?",
        "Describe transformers in AI.",
        "What is attention mechanism?",
        "Explain GPT architecture.",
        "What is BERT?",
        "Describe computer vision.",
        "What is NLP?",
        "Explain reinforcement learning."
    ]
    
    for i, topic in enumerate(topics):
        try:
            print(f"\n--- Turn {i+1}/{len(topics)} ---")
            print(f"User: {topic}")
            response = chat.send(topic)
            print(f"Assistant: {response[:150]}...")
        except Exception as e:
            warn(f"Chat failed at turn {i+1}: {e}")
            break
    
    ok("Chat history test complete")


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

    # === RUN STRESS TESTS ===
    # measure_actual_rpm(client)
    # prove_rate_limit_window(client)
    # stress_test_rate_limits(client)
    stress_test_context_window(client)
    stress_test_chat_history(client)

    # === FINAL TOKEN SUMMARY ===
    print("\n" + "=" * 70)
    ok("FINAL TOKEN SUMMARY")
    print(f"Total PROMPT tokens   : {PROMPT_TOKENS_TOTAL:,}")
    print(f"Total RESPONSE tokens : {RESPONSE_TOKENS_TOTAL:,}")
    print(f"Total tokens used     : {PROMPT_TOKENS_TOTAL + RESPONSE_TOKENS_TOTAL:,}")
    print(f"Context window limit  : {MAX_TOKENS:,}")
    percentage = ((PROMPT_TOKENS_TOTAL + RESPONSE_TOKENS_TOTAL) / MAX_TOKENS) * 100
    print(f"Percentage used       : {percentage:.2f}%")
    print("=" * 70)
    ok("All tests completed!")


if __name__ == "__main__":
    main()
