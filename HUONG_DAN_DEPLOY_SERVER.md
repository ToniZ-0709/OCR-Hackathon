# Hướng dẫn chạy hệ thống

## 1. Kết nối server
```bash
ssh -p 30945 root@103.73.232.125 -i ~/.ssh/id_rsa
```

## 2. Khởi động vLLM

Nếu vừa tạo lại `vllm_venv`, cài dependency một lần:

```bash
/root/team15/vllm_venv/bin/pip install -r /root/team15/Phase3/requirements-vllm.txt
```

Chạy:

```bash
/root/team15/Phase3/vllm_service.sh start
```

```bash
/root/team15/Phase3/vllm_service.sh logs
```

Chờ đến khi log xuất hiện:

```text
Application startup complete
```

Kiểm tra model 4B:

```bash
curl -s http://127.0.0.1:25241/v1/models
```

Kết quả phải có:

```text
fmcg-qwen3-vl-4b-lora
```

Nhấn `Ctrl+C` để thoát màn hình log.

## 3. Khởi động Streamlit

Chạy:

```bash
nohup /root/team15/Phase3/streamlit_app/start_remote.sh \
  > /root/team15/streamlit.log 2>&1 < /dev/null &
```

```bash
echo $! > /root/team15/streamlit.pid
```

```bash
tail -f /root/team15/streamlit.log
```

Chờ đến khi log xuất hiện:

```text
Uvicorn server started on 127.0.0.1:8501
```


## 4. Khởi động ngrok

Chạy:

```bash
nohup /root/ngrok http 8501 --log stdout \
  > /root/team15/ngrok.log 2>&1 < /dev/null &
```

```bash
echo $! > /root/team15/ngrok.pid
```

```bash
tail -f /root/team15/ngrok.log
```

Chờ đến khi log xuất hiện:

```text
started tunnel
```

Nhấn `Ctrl+C` để thoát màn hình log.

## 5. Mở giao diện

```text
https://delay-buffer-unnerve.ngrok-free.dev
```

## 6. Xem lại log

Log vLLM:

```bash
tail -f /root/team15/vllm.log
```

Log Streamlit:

```bash
tail -f /root/team15/streamlit.log
```

Log ngrok:

```bash
tail -f /root/team15/ngrok.log
```

Nhấn `Ctrl+C` để thoát màn hình log.

## 7. Dừng hệ thống

Dừng ngrok:

```bash
kill -TERM "$(cat /root/team15/ngrok.pid)"
```

Dừng Streamlit:

```bash
kill -TERM "$(cat /root/team15/streamlit.pid)"
```

Dừng vLLM:

```bash
/root/team15/Phase3/vllm_service.sh stop
```

Kiểm tra trạng thái vLLM:

```bash
/root/team15/Phase3/vllm_service.sh status
```
