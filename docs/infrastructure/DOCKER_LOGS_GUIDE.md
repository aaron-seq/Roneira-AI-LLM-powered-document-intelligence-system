# Docker Logs Management & Persistence

**Goal**: Persist, view, and manage application logs effectively in a containerized environment.

## 1. Viewing Logs

**Commands**:
```bash
# View all logs
docker logs [container_name]

# Follow live output
docker logs -f [container_name]

# Show last 100 lines with timestamps
docker logs --tail 100 -t [container_name]
```

## 2. Persistence Strategies

We use **Volume Mounts** for local development simplicity.

**Configuration (`docker-compose.yml`)**:
```yaml
services:
  app:
    volumes:
      - ./logs:/app/logs
```
This maps the container's log directory to your local `./logs` folder.

**Python Logging Config**:
Ensure your app writes to files in that directory:
```python
handler = RotatingFileHandler(
    '/app/logs/app.log',
    maxBytes=10*1024*1024, # 10MB
    backupCount=5
)
```

## 3. Log Rotation (Docker Native)

To preventing Docker logs from consuming all disk space:

```yaml
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 4. Production Aggregation (ELK)

For production, we recommend shipping logs to an ELK stack (Elasticsearch, Logstash, Kibana) or similar (Loki/Promtail).

**Exmaple Filebeat Config**:
1. Mount logs to Filebeat container.
2. Filebeat ships to Elasticsearch.
3. Visualize in Kibana.
