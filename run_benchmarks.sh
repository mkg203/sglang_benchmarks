source .venv/bin/activate

for i in workload_long_ctx/*;
do 
  stdbuf -oL bash run_server.sh > >(tee server.log) &
  SERVER_PID=$!

  tail -f server.log --pid=$SERVER_PID | rg -m 1 "The server is fired up and ready to roll!"

  echo "--- Server is ready. Starting Benchmarks ---"

  python benchmarks.py $i --output "results/$(echo $i  | awk -F "_turns" '{print $1}')"

  echo "--- Benchmarks finished. Shutting down server ---"
  
  kill $SERVER_PID
done
