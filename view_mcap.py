#!/usr/bin/env python3
"""
Script to view MCAP file contents in a readable format
"""
import sys
import json
from mcap.reader import make_reader

def view_mcap(filepath, max_messages=50, channel_filter=None):
    """View MCAP file contents"""
    print(f"📁 MCAP File: {filepath}\n")
    print("=" * 80)
    
    with open(filepath, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        
        # Show channels
        print("\n📡 CHANNELS:")
        channels = {}
        for channel_id, channel in summary.channels.items():
            channels[channel_id] = channel
            print(f"  [{channel_id}] {channel.topic} (schema: {channel.schema_id})")
        
        # Filter channels if specified
        if channel_filter:
            channel_ids = [cid for cid, ch in channels.items() if channel_filter in ch.topic]
        else:
            channel_ids = list(channels.keys())
        
        print(f"\n💬 MESSAGES (showing up to {max_messages}):")
        print("=" * 80)
        
        message_count = 0
        for schema, channel, message in reader.iter_messages():
            if channel.id not in channel_ids:
                continue
                
            message_count += 1
            if message_count > max_messages:
                break
            
            # Decode JSON data
            try:
                data = json.loads(message.data.decode('utf-8'))
                data_str = json.dumps(data, indent=2)
            except:
                data_str = f"<binary data: {len(message.data)} bytes>"
            
            # Format timestamp (nanoseconds to seconds)
            log_time_sec = message.log_time / 1e9 if message.log_time else 0
            pub_time_sec = message.publish_time / 1e9 if message.publish_time else 0
            
            print(f"\n[Message #{message_count}]")
            print(f"  Channel: {channel.topic} (ID: {channel.id})")
            print(f"  Log Time: {log_time_sec:.6f}s")
            print(f"  Publish Time: {pub_time_sec:.6f}s")
            print(f"  Sequence: {message.sequence}")
            print(f"  Data:")
            # Print with indentation
            for line in data_str.split('\n'):
                print(f"    {line}")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "/media/skr/storage/dreamerv4/data/vpt-owamcap/Luka-0f43d5f87f94-20220408-204320..mcap"
    max_msgs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    channel = sys.argv[3] if len(sys.argv) > 3 else None
    
    view_mcap(filepath, max_msgs, channel)
