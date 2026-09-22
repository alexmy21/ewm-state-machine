cd /home/alexmy/tools/Bonsai-demo
LD_LIBRARY_PATH="$PWD/bin/cuda" ./bin/cuda/llama-server \
    -m models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    -ngl 99 -fa on -c 2048 --host 127.0.0.1 --port 8081