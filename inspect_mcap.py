#!/usr/bin/env python3
"""
Script to inspect MCAP file contents
"""
import sys
from mcap.reader import make_reader
import json

def inspect_mcap(filepath, max_messages=20):
    """Inspect MCAP file and print summary information"""
    print(f"Inspecting MCAP file: {filepath}\n")
    print("=" * 80)
    
    try:
        with open(filepath, "rb") as f:
            reader = make_reader(f)
            
            # Print summary
            summary = reader.get_summary()
            print("\n📊 FILE SUMMARY:")
            print(f"  Schema count: {len(summary.schemas)}")
            print(f"  Channel count: {len(summary.channels)}")
            if hasattr(summary, 'attachments'):
                print(f"  Attachment count: {len(summary.attachments)}")
            if hasattr(summary, 'metadata'):
                print(f"  Metadata count: {len(summary.metadata)}")
            
            # Print schemas
            print("\n📋 SCHEMAS:")
            for schema_id, schema in reader.get_summary().schemas.items():
                print(f"  Schema ID {schema_id}:")
                print(f"    Name: {schema.name}")
                print(f"    Encoding: {schema.encoding}")
                print(f"    Data length: {len(schema.data)} bytes")
                if len(schema.data) < 500:  # Print if not too large
                    print(f"    Data preview: {schema.data[:200]}...")
            
            # Print channels
            print("\n📡 CHANNELS:")
            for channel_id, channel in reader.get_summary().channels.items():
                print(f"  Channel ID {channel_id}:")
                print(f"    Topic: {channel.topic}")
                print(f"    Schema ID: {channel.schema_id}")
                print(f"    Message encoding: {channel.message_encoding}")
                print(f"    Metadata: {channel.metadata}")
            
            # Print metadata
            if hasattr(summary, 'metadata') and summary.metadata:
                print("\n📝 METADATA:")
                for metadata in summary.metadata:
                    print(f"  {metadata.name}:")
                    print(f"    {metadata.metadata}")
            
            # Read and print sample messages
            print("\n💬 SAMPLE MESSAGES (first {}):".format(max_messages))
            print("-" * 80)
            message_count = 0
            for schema, channel, message in reader.iter_messages():
                message_count += 1
                if message_count > max_messages:
                    break
                    
                print(f"\nMessage #{message_count}:")
                print(f"  Channel: {channel.topic} (ID: {channel.id})")
                print(f"  Timestamp: {message.log_time} (log) / {message.publish_time} (publish)")
                print(f"  Sequence: {message.sequence}")
                print(f"  Data size: {len(message.data)} bytes")
                
                # Try to decode if it's JSON or text
                try:
                    decoded = message.data.decode('utf-8')
                    if len(decoded) < 500:
                        print(f"  Data preview: {decoded[:200]}...")
                    else:
                        print(f"  Data preview: {decoded[:200]}... (truncated)")
                except:
                    print(f"  Data: <binary, {len(message.data)} bytes>")
            
            print(f"\n\nTotal messages in file: {message_count} (showing first {max_messages})")
            
    except Exception as e:
        print(f"Error reading MCAP file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "/media/skr/storage/dreamerv4/data/vpt-owamcap/Luka-0f43d5f87f94-20220408-204320..mcap"
    max_msgs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    inspect_mcap(filepath, max_msgs)
