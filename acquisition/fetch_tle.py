import requests
import psycopg2
from datetime import datetime, timezone
import os
import time
import sys

# Environment variables
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_USER = os.getenv('DB_USER', 'admin')
DB_PASS = os.getenv('DB_PASS', 'admin')
DB_NAME = os.getenv('DB_NAME', 'satellites')

# TLE sources - multiple URLs for redundancy
TLE_SOURCES = [
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",
]

def connect_db(max_retries=5):
    """Connect to database with retry logic"""
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASS
            )
            print(f"✅ Connected to database at {DB_HOST}")
            return conn
        except Exception as e:
            print(f"⏳ Database connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise

def is_valid_tle_line(line, line_number):
    """Validate TLE line format"""
    line = line.strip()
    
    # TLE line 1 must start with '1' and be 69 chars
    if line_number == 1:
        return line.startswith('1 ') and len(line) == 69
    
    # TLE line 2 must start with '2' and be 69 chars
    if line_number == 2:
        return line.startswith('2 ') and len(line) == 69
    
    return False

def fetch_tle_data(url):
    """Fetch TLE data from URL with validation"""
    try:
        print(f"📡 Fetching from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if we got HTML instead of TLE
        content_type = response.headers.get('Content-Type', '')
        if 'html' in content_type.lower():
            print(f"⚠️  Got HTML instead of TLE data from {url}")
            return []
        
        text = response.text.strip()
        
        # Check if response looks like TLE data
        if '<!DOCTYPE' in text or '<html' in text:
            print(f"⚠️  Response contains HTML, skipping {url}")
            return []
        
        lines = text.split('\n')
        
        # Parse TLE data (3 lines per satellite: name, line1, line2)
        tle_list = []
        i = 0
        
        while i < len(lines) - 2:
            sat_name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
            
            # Skip empty lines
            if not sat_name or not line1 or not line2:
                i += 1
                continue
            
            # Validate TLE format
            if is_valid_tle_line(line1, 1) and is_valid_tle_line(line2, 2):
                tle_list.append({
                    'name': sat_name[:255],  # Limit to 255 chars
                    'line1': line1,
                    'line2': line2
                })
                i += 3
            else:
                # Not valid TLE, try next line
                i += 1
        
        print(f"✅ Found {len(tle_list)} valid TLE records")
        return tle_list
        
    except requests.RequestException as e:
        print(f"❌ Failed to fetch from {url}: {e}")
        return []
    except Exception as e:
        print(f"❌ Error parsing TLE data: {e}")
        return []

def insert_tle_data(conn, tle_list):
    """Insert TLE data into database"""
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    
    try:
        for tle in tle_list:
            try:
                # Use timezone-aware datetime
                now = datetime.now(timezone.utc)
                
                cur.execute("""
                    INSERT INTO tle_data (satellite_name, line1, line2, timestamp) 
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (satellite_name, line1, line2) DO NOTHING
                """, (tle['name'], tle['line1'], tle['line2'], now))
                
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                print(f"⚠️  Failed to insert {tle['name']}: {e}")
                continue
        
        conn.commit()
        print(f"💾 Inserted: {inserted}, Skipped duplicates: {skipped}")
        return inserted
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Database error: {e}")
        raise
    finally:
        cur.close()

def main():
    """Main acquisition loop"""
    print("=" * 60)
    print("🛰️  Satellite TLE Acquisition Service")
    print("=" * 60)
    
    # Connect to database
    try:
        conn = connect_db()
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)
    
    # Fetch TLE data from all sources
    total_inserted = 0
    all_tle_data = []
    
    for url in TLE_SOURCES:
        tle_list = fetch_tle_data(url)
        all_tle_data.extend(tle_list)
        time.sleep(2)  # Be nice to the server
    
    print(f"\n📊 Total TLE records collected: {len(all_tle_data)}")
    
    # Insert into database
    if all_tle_data:
        total_inserted = insert_tle_data(conn, all_tle_data)
        print(f"✅ Successfully inserted {total_inserted} new TLE records")
    else:
        print("⚠️  No TLE data collected")
    
    # Show statistics
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT satellite_name, COUNT(*) as count 
            FROM tle_data 
            GROUP BY satellite_name 
            ORDER BY count DESC 
            LIMIT 10
        """)
        
        print(f"\n📈 Top 10 satellites by TLE count:")
        for row in cur.fetchall():
            print(f"   {row[0]}: {row[1]} records")
        
        cur.execute("SELECT COUNT(DISTINCT satellite_name) FROM tle_data")
        total_sats = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM tle_data")
        total_tle = cur.fetchone()[0]
        
        print(f"\n📊 Database Statistics:")
        print(f"   Total satellites: {total_sats}")
        print(f"   Total TLE records: {total_tle}")
        
        cur.close()
        
    except Exception as e:
        print(f"⚠️  Failed to get statistics: {e}")
    
    # Close connection
    conn.close()
    print("\n✅ Acquisition complete!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Acquisition stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)