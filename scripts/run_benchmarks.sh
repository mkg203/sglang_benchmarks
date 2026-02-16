# NOTE: RUN THIS FILE FROM ROOT DIR
#!/bin/bash

source .venv/bin/activate

# --- CONFIGURATION ---
SERVER_PORT=30000
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct" 
# ---------------------

ENABLE_CPU_CACHE=0
for arg in "$@"; do
    if [[ "$arg" == "--enable-cpu-cache" ]]; then
        ENABLE_CPU_CACHE=1
        echo "CPU Hierarchical Cache ENABLED"
    fi
done

cleanup() {
    echo ""
    echo "!!! CAUGHT EXIT SIGNAL / CLEANING UP !!!"
    
    if [[ -n "$SERVER_PID" ]]; then
        echo "Killing process group $SERVER_PID..."
        kill -TERM -"$SERVER_PID" 2>/dev/null 
        kill "$SERVER_PID" 2>/dev/null
    fi

    fuser -k -TERM "$SERVER_PORT/tcp" >/dev/null 2>&1
    pkill -f "sglang" 2>/dev/null
    
    sleep 3
    echo "Cleanup complete."
}

trap 'cleanup; exit 1' SIGINT SIGTERM

mkdir -p results

CACHE_FLAG=""
if [[ "$ENABLE_CPU_CACHE" -eq 1 ]]; then
    CACHE_FLAG="--enable-hierarchical-cache"
fi

# --- MAIN LOOP ---
for i in workload_long_ctx/*; do
    [ -e "$i" ] || continue
    
    FILENAME=$(basename "$i")
    OUTPUT_NAME="${FILENAME%%_turns*}"
    
    echo "=================================="
    echo "Processing Workload: $OUTPUT_NAME"
    
    > server.log


    echo "Starting SGLang Server..."

    setsid stdbuf -oL python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$SERVER_PORT" \
        --enable-metrics \
        --mem-fraction-static 0.4 \
        $CACHE_FLAG > server.log 2>&1 &
        
    SERVER_PID=$!
    echo "Server Process Group: $SERVER_PID"

    SERVER_READY=0
    MAX_RETRIES=75 
    COUNT=0
    
    while [ $COUNT -lt $MAX_RETRIES ]; do
        if grep -q "The server is fired up and ready to roll!" server.log; then
            SERVER_READY=1
            break
        fi
        
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server process died unexpectedly!"
            break
        fi
        
        sleep 1
        ((COUNT++))
    done

    if [[ $SERVER_READY == 1 ]]; then
        echo "--- Server Ready. Running Benchmark ---"
        
        python -m src.benchmarks "$i" --output "$OUTPUT_NAME"
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -ne 0 ]; then
             echo "!!! ERROR: Benchmark failed with code $EXIT_CODE !!!"
             cleanup 
             exit 1
        fi

    else
        echo "!!! ERROR: Server failed to start (Timeout or Crash) !!!"
        cleanup
        exit 1
    fi

    echo "Finished $OUTPUT_NAME. Stopping server..."
    cleanup
    
    SERVER_PID="" 
    
    echo "----------------------------------"
done
