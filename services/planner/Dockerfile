FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY avoid/ /app/avoid/
COPY server.py .
EXPOSE 8060
CMD ["uvicorn","server:svc","--host","0.0.0.0","--port","8060"]
