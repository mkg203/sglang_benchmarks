source .venv/bin/activate

for i in workload_long_ctx/*;
do 
  > server.log
  stdbuf -oL bash run_server.sh > >(tee server.log) &
  SERVER_PID=$!

  tail -f server.log --pid=$SERVER_PID | grep -m 1 "/generate"
  sleep 5

  echo "--- Server is ready. Starting Benchmarks ---"

  python benchmarks.py $i --output "results/$(echo $i  | awk -F "_turns" '{print $1}')" > benchmark_progress.log

  echo "--- Benchmarks finished. Shutting down server ---"
  
  kill $SERVER_PID
done
