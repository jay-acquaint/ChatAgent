import redis
import os

# Connect to Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    db=0,
    decode_responses=True
)


def get_history(session_id: str):
    key = f"chat:{session_id}"
    return redis_client.lrange(key, 0, -1)


def save_message(session_id: str, message: str):
    key = f"chat:{session_id}"

    redis_client.rpush(key, message)

    # Keep only last 20 messages
    redis_client.ltrim(key, -20, -1)

    # auto-expire after 24 hour
    redis_client.expire(key, 86400)