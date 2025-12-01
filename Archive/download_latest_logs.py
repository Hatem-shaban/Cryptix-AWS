#!/usr/bin/env python3
"""
Download latest trading logs from Render deployment
"""

import requests
import zipfile
import io
import os
from pathlib import Path
from datetime import datetime

def download_logs_from_render():
    """Download the latest logs from the Render deployment"""
    
    # Your Render app URL
    render_url = "https://cryptix-6yol.onrender.com"
    download_endpoint = f"{render_url}/download_logs"
    
    print("🚀 Downloading latest logs from Render...")
    print(f"   URL: {download_endpoint}")
    
    try:
        # Download the zip file
        response = requests.get(download_endpoint, timeout=30)
        response.raise_for_status()
        
        # Check if we got a zip file
        if 'application/zip' not in response.headers.get('content-type', ''):
            print("❌ Error: Response is not a zip file")
            print(f"   Content-Type: {response.headers.get('content-type')}")
            return False
        
        # Extract the zip file
        zip_content = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_content, 'r') as zip_ref:
            # List files in the zip
            files_in_zip = zip_ref.namelist()
            print(f"📦 Files in download: {', '.join(files_in_zip)}")
            
            # Create logs directory if it doesn't exist
            logs_dir = Path('logs')
            logs_dir.mkdir(exist_ok=True)
            
            # Extract all files to logs directory
            extracted_files = []
            for file_name in files_in_zip:
                if file_name.endswith('.csv') or file_name.endswith('.json') or file_name.endswith('.log'):
                    # Extract to logs directory
                    source = zip_ref.read(file_name)
                    target_path = logs_dir / file_name
                    
                    # Backup existing file if it exists
                    if target_path.exists():
                        backup_path = target_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                        target_path.rename(backup_path)
                        print(f"   📋 Backed up existing {file_name} to {backup_path.name}")
                    
                    # Write new file
                    with open(target_path, 'wb') as f:
                        f.write(source)
                    
                    extracted_files.append(file_name)
                    print(f"   ✅ Updated: {file_name}")
            
            if extracted_files:
                print(f"\n🎉 Successfully downloaded and updated {len(extracted_files)} files!")
                print("   Files updated:")
                for file_name in extracted_files:
                    file_path = logs_dir / file_name
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        print(f"     • {file_name} ({file_size:,} bytes)")
                return True
            else:
                print("❌ No files were extracted from the zip")
                return False
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error downloading logs: {e}")
        return False
    except zipfile.BadZipFile as e:
        print(f"❌ Invalid zip file: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_local_files():
    """Check the status of local CSV files"""
    logs_dir = Path('logs')
    csv_files = ['trade_history.csv', 'signal_history.csv', 'error_log.csv']
    
    print("\n📊 Local CSV file status:")
    for csv_file in csv_files:
        file_path = logs_dir / csv_file
        if file_path.exists():
            file_size = file_path.stat().st_size
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"   • {csv_file}: {file_size:,} bytes (modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print(f"   • {csv_file}: ❌ Not found")

def main():
    """Main function"""
    print("=" * 60)
    print("📥 CRYPTIX Log Downloader")
    print("=" * 60)
    
    # Check current state
    check_local_files()
    
    # Download latest logs
    success = download_logs_from_render()
    
    if success:
        print("\n" + "=" * 60)
        check_local_files()
        print("=" * 60)
        print("✅ Log download completed successfully!")
        print("📋 Your local CSV files now contain the latest trading data from Render.")
    else:
        print("\n❌ Failed to download logs. Please check your connection and try again.")

if __name__ == "__main__":
    main()