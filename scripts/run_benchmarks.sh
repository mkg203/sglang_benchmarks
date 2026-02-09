#!/bin/bash

source .venv/bin/activate

# --- CONFIGURATION ---
SERVER_PORT=30000
# ---------------------

cleanup() {
    echo "CLEANING UP"
    
    if [[ -n "$SERVER_PID" ]]; then
        echo "Killing wrapper script (PID $SERVER_PID)..."
        kill -TERM -"$SERVER_PID" 2>/dev/null 
        kill "$SERVER_PID" 2>/dev/null
    fi

    echo "Ensuring sglang python processes are dead..."
    
    fuser -k -TERM "$SERVER_PORT/tcp" >/dev/null 2>&1
    
    pkill -f "python3 -m sglang.launch_server" 2>/dev/null
    
    sleep 2
    
    echo "Cleanup complete."
}

trap 'echo "!!! CAUGHT EXIT SIGNAL !!!"
; cleanup; exit 1' SIGINT SIGTERM

mkdir -p results

for i in workload_long_ctx/*; do
    [ -e "$i" ] || continue
    
    FILENAME=$(basename "$i")
    OUTPUT_NAME="${FILENAME%%_turns*}"
    
    echo "=================================="
    echo "Processing: $OUTPUT_NAME"
    
    > server.log

    setsid stdbuf -oL bash run_server.sh > server.log 2>&1 &
    SERVER_PID=$!
    
    echo "Server Process Group: $SERVER_PID"

    SERVER_READY=0
    MAX_RETRIES=60
    COUNT=0
    
    while [ $COUNT -lt $MAX_RETRIES ]; do
        if grep -q "The server is fired up and ready to roll!" server.log; then
            SERVER_READY=1
            break
        fi
        
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server crashed before becoming ready!"
            break
        fi
        
        sleep 1
        ((COUNT++))
    done

    if [[ $SERVER_READY == 1 ]]; then
        echo "--- Server is Ready. Running Benchmark ---"
        
        python -m src.benchmarks.py "$i" --output "results/$OUTPUT_NAME"
        # sleep 5 
        
        if [ $? -ne 0 ]; then
             echo "!!! ERROR: Benchmark script failed for $OUTPUT_NAME !!!"
             cleanup
             exit 1
        fi

    else
        echo "!!! ERROR: Server failed to start or timed out for $OUTPUT_NAME !!!"
        cat server.log
        cleanup
        exit 1
    fi

    echo "Finished $OUTPUT_NAME. Stopping server..."
    cleanup
    
    SERVER_PID="" 
    
    echo "----------------------------------"
done
