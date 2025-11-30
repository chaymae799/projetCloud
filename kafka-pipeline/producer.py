






from kafka import KafkaProducer
import psycopg2
import json
import time
import os

KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:9092')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres-db')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'satellites')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'admin')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'admin')

print(f"🔌 Connecting to Kafka at {KAFKA_BROKER}")
print(f"🔌 Connecting to PostgreSQL at {POSTGRES_HOST}")

producer = None
for attempt in range(10):
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            api_version=(0, 10, 1),
            max_block_ms=60000
        )
        print("✅ Connected to Kafka")
        break
    except Exception as e:
        print(f"⚠️  Kafka connection attempt {attempt + 1}/10 failed: {e}")
        time.sleep(5)

if not producer:
    print("❌ Failed to connect to Kafka after 10 attempts")
    exit(1)

print("🔌 Connecting to database...")
conn = psycopg2.connect(
    host=POSTGRES_HOST,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
)
print("✅ Connected to PostgreSQL")

print("🚀 Starting real-time TLE streaming...")
print("📡 Sending TLE data every 60 seconds")

while True:
    try:
        cur = conn.cursor()
        
        cur.execute("""
            SELECT DISTINCT ON (satellite_name) 
                satellite_name, line1, line2, timestamp 
            FROM tle_data 
            ORDER BY satellite_name, timestamp DESC
        """)
        
        satellites = cur.fetchall()
        print(f"\n📊 Found {len(satellites)} satellites to process")
        
        for row in satellites:
            message = {
                'satellite_name': row[0],
                'line1': row[1],
                'line2': row[2],
                'timestamp': row[3].isoformat()
            }
            
            producer.send('tle-raw', message)
            print(f"  ✅ Sent: {row[0]}")
        
        producer.flush()
        cur.close()
        
        print(f"⏰ Waiting 60 seconds before next batch...")
        time.sleep(60)
        
    except Exception as e:
        print(f"❌ Error in producer loop: {e}")
        time.sleep(10)