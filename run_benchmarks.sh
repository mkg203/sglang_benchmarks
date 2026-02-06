for i in $(ls workload_long_ctx);
do 
  bash run_server.sh > >(tee server.log) &
  SERVER_PID=$!

  tail -f server.log | rg -m 1 "READY"

  echo "--- Server is ready. Starting Benchmarks ---"

  python benchmarks.py $i --output "results/$(echo $i  | awk -F "_turns" '{print $1}')"

  echo "--- Benchmarks finished. Shutting down server ---"
  
  kill $SERVER_PID
done
