#!/bin/bash
# Start CRYPTIX-ML Bot with Python 3.11
# This ensures Supabase and all modern dependencies work correctly

cd /home/opc

# Kill any existing Python processes
pkill -f "python.*render_launcher" 2>/dev/null
pkill -f "python.*web_bot" 2>/dev/null

# Wait for processes to terminate
sleep 2

# Start with Python 3.11
echo "Starting CRYPTIX-ML Bot with Python 3.11..."
/usr/bin/python3.11 render_launcher.py > bot.log 2>&1 &

echo "Bot started with PID: $!"
echo "Logs available at /home/opc/bot.log"
