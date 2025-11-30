
from kafka import KafkaConsumer, KafkaProducer
import requests
import json
import time
import os

KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:9092')
AI_SERVICE_URL = 'http://orbital-ai:8000'

print(f"🔌 Connecting to Kafka at {KAFKA_BROKER}")
print(f"🔌 AI Service at {AI_SERVICE_URL}")

# Wait for Kafka to be ready
print("⏳ Waiting 10 seconds for Kafka...")
time.sleep(10)

# Initialize Kafka consumer
print("📥 Initializing Kafka consumer...")
consumer = KafkaConsumer(
    'tle-raw',
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    group_id='processor-group',
    api_version=(0, 10, 1)
)

# Initialize Kafka producer for predictions
print("📤 Initializing Kafka producer...")
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    api_version=(0, 10, 1)
)

print("✅ Kafka consumer and producer ready")
print("🚀 Starting real-time prediction processor...")
print("⏳ Waiting for messages from 'tle-raw' topic...\n")

message_count = 0

for message in consumer:
    try:
        tle_data = message.value
        satellite_name = tle_data['satellite_name']
        
        print(f"\n📡 [{message_count + 1}] Processing: {satellite_name}")
        
        # Call AI service for prediction
        response = requests.post(
            f'{AI_SERVICE_URL}/predict',
            json={
                'satellite_name': satellite_name,
                'sequence_length': 5
            },
            timeout=10
        )
        
        if response.status_code == 200:
            prediction = response.json()
            
            # Extract first prediction for display
            first_pred = prediction['predictions'][0]
            print(f"  ✅ Predicted next position:")
            print(f"     X: {first_pred['position_x_km']:.2f} km")
            print(f"     Y: {first_pred['position_y_km']:.2f} km")
            print(f"     Z: {first_pred['position_z_km']:.2f} km")
            
            # Send to predictions topic
            output_message = {
                'satellite_name': satellite_name,
                'predictions': prediction['predictions'],
                'model_info': prediction['model_info'],
                'timestamp': prediction['generated_at']
            }
            
            producer.send('predictions', output_message)
            producer.flush()
            
            print(f"  📤 Published to 'predictions' topic")
            
        elif response.status_code == 404:
            print(f"  ⚠️  No model trained for {satellite_name} yet")
            
        else:
            print(f"  ❌ Error: {response.status_code} - {response.text}")
        
        message_count += 1
        
    except requests.exceptions.Timeout:
        print(f"  ⏱️  Timeout calling AI service")
    except requests.exceptions.ConnectionError:
        print(f"  🔌 Cannot connect to AI service at {AI_SERVICE_URL}")
    except Exception as e:
        print(f"  ❌ Error: {e}")