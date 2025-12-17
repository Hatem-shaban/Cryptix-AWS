# AWS EC2 Troubleshooting Guide

## Common Issues & Solutions

---

## 🔴 CONNECTIVITY ISSUES

### Issue 1: "Connection timed out" when SSHing

**Error Message:**
```
ssh -i "cryptix-ec2.pem" ubuntu@54.194.123.45
ssh: connect to host 54.194.123.45 port 22: Connection timed out
```

**Possible Causes & Solutions:**

**Cause 1: Instance Still Booting (Most Common)**
- ✅ **Solution**: Wait 3-5 minutes after instance creation
- Status checks should show "2/2 passed" before connecting
- View: EC2 Dashboard > Instances > Click your instance > Status Checks

**Cause 2: Security Group Missing SSH Rule**
- ✅ **Solution**: 
  ```
  1. Go to EC2 Dashboard > Security Groups
  2. Find 'cryptix-sg'
  3. Click 'Inbound Rules'
  4. Verify SSH (port 22) rule exists with Source: 0.0.0.0/0
  5. If missing, click 'Edit Inbound Rules' and add:
     - Type: SSH
     - Protocol: TCP
     - Port: 22
     - Source: 0.0.0.0/0
  ```

**Cause 3: SSH Key Permissions Wrong**
- ✅ **Solution** (Windows PowerShell):
  ```powershell
  # Check current permissions
  icacls "cryptix-ec2.pem"
  
  # Fix permissions
  icacls "cryptix-ec2.pem" /inheritance:r
  icacls "cryptix-ec2.pem" /grant:r "$env:USERNAME`:`(F`)"
  
  # Verify ownership
  icacls "cryptix-ec2.pem"
  ```

**Cause 4: Network Interface Issue**
- ✅ **Solution**:
  ```
  1. Go to EC2 Dashboard > Instances
  2. Right-click instance > Instance State > Reboot
  3. Wait 2 minutes for reboot
  4. Try SSH again
  ```

---

### Issue 2: "Permission denied (publickey)"

**Error Message:**
```
ubuntu@54.194.123.45: Permission denied (publickey)
```

**Possible Causes & Solutions:**

**Cause 1: Wrong SSH Key File**
- ✅ **Solution**: Use the EXACT key file you downloaded from AWS
  ```bash
  # Verify you have the right file
  ls -la cryptix-ec2.pem
  
  # If wrong file, download correct one from AWS:
  # EC2 Dashboard > Key Pairs > Download new .pem file
  ```

**Cause 2: Wrong Username**
- ✅ **Solution**: Use `ubuntu` NOT `opc` or `admin`
  ```bash
  # ✓ Correct
  ssh -i "cryptix-ec2.pem" ubuntu@54.194.123.45
  
  # ✗ Wrong
  ssh -i "cryptix-ec2.pem" opc@54.194.123.45
  ssh -i "cryptix-ec2.pem" admin@54.194.123.45
  ```

**Cause 3: Key Not in Instance**
- ✅ **Solution**: Regenerate key pair and instance
  ```
  1. EC2 Dashboard > Key Pairs > Delete old pair
  2. EC2 Dashboard > Instances > Terminate old instance
  3. Create NEW instance and select NEW key pair
  4. Try SSH with NEW key
  ```

---

### Issue 3: "No such file or directory"

**Error Message:**
```powershell
ssh: cryptix-ec2.pem: No such file or directory
```

**Cause**: File path is wrong

**Solution**:
```powershell
# Navigate to where you downloaded the file
cd $env:USERPROFILE\Downloads

# List files
ls *.pem

# Try SSH from correct directory
ssh -i "cryptix-ec2.pem" ubuntu@54.194.123.45

# Or use full path
ssh -i "$env:USERPROFILE\Downloads\cryptix-ec2.pem" ubuntu@54.194.123.45
```

---

## 🔴 INSTALLATION & DEPENDENCIES

### Issue 4: "pip install fails" with wheel compilation errors

**Error Message:**
```
error: subprocess-exited-with-error
error: Microsoft Visual C++ 14.0 or greater is required
```

**Cause**: Missing C++ compiler for Windows subsystem on AWS

**Solution 1 (Recommended - Skip problematic packages):**
```bash
# Install without building from source
pip install -r requirements.txt --only-binary :all:

# Or skip specific packages and install individually
pip install --no-build-isolation -r requirements.txt
```

**Solution 2 (Install pre-built wheels):**
```bash
# Check if Ubuntu has pre-built packages
sudo apt install python3-numpy python3-scipy python3-sklearn python3-pandas -y
pip install -r requirements.txt --system-site-packages
```

**Solution 3 (Install build tools on Ubuntu):**
```bash
sudo apt install -y build-essential python3-dev libssl-dev libffi-dev
pip install -r requirements.txt
```

---

### Issue 5: "No module named 'sklearn'" or similar import errors

**Error Message:**
```
ModuleNotFoundError: No module named 'sklearn'
```

**Cause**: Not in virtual environment OR packages not installed

**Solution**:
```bash
# Verify you're in venv (should show (venv) in prompt)
source ~/Cryptix-AWS/venv/bin/activate
# Output should be: (venv) ubuntu@...

# Reinstall requirements
pip install -r requirements.txt

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

---

## 🔴 BOT STARTUP ISSUES

### Issue 6: Systemd service won't start

**Error Message:**
```bash
$ sudo systemctl status cryptix
● cryptix.service - CRYPTIX Trading Bot
  Loaded: loaded
  Active: failed (Result: exit-code)
```

**Debug Steps:**
```bash
# View detailed error
sudo journalctl -u cryptix -n 100 --no-pager

# Test running the bot manually
cd ~/Cryptix-AWS
source venv/bin/activate
python render_launcher.py

# This will show the actual error message
```

**Common Causes & Solutions:**

**Cause 1: Missing .env file or invalid values**
```bash
# Check .env exists and is readable
cat ~/.env | head -10

# Verify required variables
grep "API_KEY\|TELEGRAM_BOT_TOKEN" ~/.env

# If missing, edit and restart
nano ~/.env
sudo systemctl restart cryptix
```

**Cause 2: Port 5000 already in use**
```bash
# Find what's using port 5000
sudo lsof -i :5000

# If another process is using it, kill it
sudo kill -9 <PID>

# Restart bot
sudo systemctl restart cryptix
```

**Cause 3: Permission issues**
```bash
# Fix directory permissions
sudo chown -R ubuntu:ubuntu ~/Cryptix-AWS
chmod -R 755 ~/Cryptix-AWS
chmod -R 644 ~/Cryptix-AWS/.*

# Restart
sudo systemctl restart cryptix
```

**Cause 4: Python venv broken**
```bash
# Recreate virtual environment
cd ~/Cryptix-AWS
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Restart service
sudo systemctl restart cryptix
```

---

### Issue 7: "Bot crashes immediately after starting"

**Error Pattern:**
```
Active: active (running)  [then after 10 seconds]
Active: failed (Result: exit-code) Process exited with code...
```

**Debug:**
```bash
# Check full logs
sudo journalctl -u cryptix -n 200 --no-pager

# Run manually to see error
cd ~/cryptix
source venv/bin/activate
python render_launcher.py

# Look for error in output (may be 30-50 lines)
```

**Common Issues:**

1. **API Credentials Invalid**
   ```
   Error: Invalid API key
   ```
   ✅ Fix: Check Binance API key in .env

2. **Binance API Unavailable**
   ```
   Error: Connection refused / Network unreachable
   ```
   ✅ Fix: Check internet connection, Binance status

3. **Supabase Connection Failed**
   ```
   Error: Failed to connect to supabase
   ```
   ✅ Fix: Check SUPABASE_URL and SUPABASE_SERVICE_KEY in .env

4. **Telegram Token Invalid**
   ```
   Error: Invalid Telegram token
   ```
   ✅ Fix: Update .env with correct token (non-critical)

---

## 🔴 RUNTIME ISSUES

### Issue 8: Out of memory (OOM) - Bot killed

**Error Message:**
```
Killed
Process: cryptix
Signal: 9 (SIGKILL) - out of memory
```

**Cause**: t2.micro has only 1GB RAM, sometimes not enough

**Solutions:**

**Solution 1 (Immediate): Add Swap Space**
```bash
# Create 2GB swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify swap
free -h
# Output should show 2G swap

# Make permanent (survives reboot)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Solution 2 (Optimize Bot):**
```bash
# Edit config to reduce memory usage
nano ~/cryptix/config.py

# Look for and adjust:
LOOKBACK_PERIOD = 100  # Reduce from 1000 or higher
BATCH_SIZE = 50        # Reduce from 100+
MAX_CACHE_SIZE = 100   # Reduce cache

sudo systemctl restart cryptix
```

**Solution 3 (Monitor Memory):**
```bash
# Watch memory in real-time
watch -n 1 free -h

# View what's using memory
ps aux --sort=-%mem | head -10

# Check specific process
ps aux | grep python
```

---

### Issue 9: High CPU usage / Bot running slow

**Check CPU:**
```bash
# View current CPU
top
# Press 'q' to exit

# Check load average
uptime
# Output: load average: 0.5, 0.4, 0.3  (values < 2 are OK for dual core)
```

**t2.micro CPU Burst Explained:**
- **Baseline**: 20% CPU (earn credits)
- **Burst**: Up to 100% CPU (use credits)
- **Unlimited**: Can go over limit (small charges apply)

**Solutions:**

**Check if CPU limited:**
```bash
# View detailed metrics
cat /sys/devices/system/cpu/cpu_throttling_states  # May not exist

# Monitor with htop (better than top)
sudo apt install -y htop
htop
```

**Reduce CPU usage:**
1. **Disable intensive features** in config.py
2. **Reduce data lookback** (less historical data = less processing)
3. **Increase trade check interval** (check less frequently)

---

### Issue 10: "Connection refused" to Binance/Telegram/Supabase

**Error Message:**
```
ERROR: Failed to connect to api.binance.com
ERROR: Connection refused / Network unreachable
```

**Cause**: Internet connectivity issue

**Debug:**
```bash
# Test internet connectivity
ping 8.8.8.8      # Google DNS
ping api.binance.com
curl https://api.binance.com/api/v3/time

# Check DNS
nslookup api.binance.com
```

**Solutions:**

**If ping fails:**
- ✅ Your EC2 instance has no internet
- ✅ Check: EC2 > Instance > Networking > Network interfaces > Subnet > Route table
- ✅ Verify route: 0.0.0.0/0 → Internet Gateway

**If curl fails:**
- ✅ Firewall blocking outbound traffic
- ✅ Check: Security Group > Outbound Rules > Should allow all (0.0.0.0/0)

**If nslookup fails:**
- ✅ DNS resolution problem
- ✅ Edit `/etc/resolv.conf`:
  ```bash
  sudo nano /etc/resolv.conf
  # Add: nameserver 8.8.8.8
  ```

---

## 🔴 WEB DASHBOARD ISSUES

### Issue 11: "Connection refused" when accessing http://instance-ip

**Error Message (in browser):**
```
This site can't be reached
Connection refused
```

**Debug:**
```bash
# Check if Flask is running
ps aux | grep render_launcher
ps aux | grep python

# Check if port 5000 is listening
sudo lsof -i :5000

# Test locally on instance
curl http://localhost:5000
curl http://localhost:5000/health

# Check Nginx
sudo systemctl status nginx
sudo tail -f /var/log/nginx/error.log
```

**Solutions:**

**If Flask not running:**
```bash
# Start service
sudo systemctl start cryptix
sudo systemctl status cryptix

# Check errors
sudo journalctl -u cryptix -n 50 --no-pager
```

**If port 5000 not listening:**
```bash
# Verify systemd file paths are correct
grep "ExecStart" /etc/systemd/system/cryptix.service
# Should show: /home/ubuntu/Cryptix-AWS/venv/bin/python render_launcher.py

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart cryptix
```

**If Nginx not working:**
```bash
# Test Nginx config
sudo nginx -t
# Should output: nginx: configuration is ok

# Check error log
sudo tail -f /var/log/nginx/error.log

# Verify proxy_pass is correct
grep "proxy_pass" /etc/nginx/sites-available/cryptix
# Should show: proxy_pass http://127.0.0.1:5000;

# Restart
sudo systemctl restart nginx
```

**If security group blocks traffic:**
```
1. Go to EC2 Dashboard > Security Groups > cryptix-sg
2. Click Inbound Rules
3. Verify HTTP (80) rule: Type=HTTP, Port=80, Source=0.0.0.0/0
4. Verify Custom TCP (5000): Type=Custom TCP, Port=5000, Source=0.0.0.0/0
5. If missing, add them
```

---

### Issue 12: Dashboard loads but shows errors

**Symptoms**: Page loads but shows "ERROR" or blank content

**Debug:**
```bash
# Check Flask logs
sudo journalctl -u cryptix -f

# Check web server logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Check application logs
tail -f ~/cryptix/logs/error_log.csv
```

**Common Issues:**

1. **Missing data files**
   ```
   Check if models/ and logs/ directories exist
   ls -la ~/Cryptix-AWS/models/
   ls -la ~/Cryptix-AWS/logs/
   ```

2. **Supabase disconnected**
   ```
   Verify SUPABASE_URL and SUPABASE_SERVICE_KEY
   grep SUPABASE ~/.env
   ```

3. **Permissions issue**
   ```bash
   # Fix directory permissions
   sudo chown -R ubuntu:ubuntu ~/cryptix/logs
   sudo chmod -R 755 ~/cryptix/logs
   ```

---

## 🔴 PERFORMANCE ISSUES

### Issue 13: Trades executing very slowly

**Cause**: High latency to Binance or CPU throttling

**Check latency:**
```bash
# Ping Binance
ping api.binance.com
# Should be < 100ms

# Check from multiple regions
curl -w "@curl-format.txt" -o /dev/null -s https://api.binance.com/api/v3/time
```

**Solutions:**
1. **Change AWS region** to closer to Binance servers
   - Binance is in Asia, so Ireland/Frankfurt/Singapore recommended
   
2. **Check CPU throttling:**
   ```bash
   # If running on burst credits, may slow down
   aws ec2 describe-instances --instance-ids <instance-id>
   # Look for "cpuCredits"
   ```

3. **Upgrade instance** (after free tier ends)
   ```
   t2.small has consistent CPU performance vs t2.micro burst
   ```

---

### Issue 14: Logs growing too fast (filling disk)

**Check disk usage:**
```bash
df -h
du -sh ~/cryptix/logs/
```

**Solutions:**

**Immediate: Delete old logs**
```bash
# Delete logs older than 30 days
find ~/cryptix/logs -name "*.csv" -mtime +30 -delete
find ~/cryptix/logs -name "*.txt" -mtime +30 -delete

# Check freed space
df -h
```

**Long-term: Compress logs**
```bash
# Compress old logs
gzip ~/cryptix/logs/error_log*.csv
gzip ~/cryptix/logs/ml_training_data*.txt

# Check compression
ls -lh ~/cryptix/logs/
# .csv.gz files use 90% less space
```

**Alternative: Reduce logging verbosity**
```bash
# Edit config
nano ~/cryptix/config.py

# Find LOG_LEVEL or similar settings
# Reduce logging detail
```

---

## ✅ VERIFICATION CHECKLIST

After troubleshooting, verify everything is working:

```bash
# 1. SSH connection works
ssh -i "cryptix-ec2.pem" ubuntu@YOUR_IP

# 2. Systemd service running
sudo systemctl status cryptix
# Expected: Active (running)

# 3. Flask responding
curl http://localhost:5000/health
# Expected: JSON response

# 4. Nginx working
curl http://localhost
# Expected: HTML response

# 5. Memory OK
free -h
# Expected: Used < 700MB

# 6. No errors in logs
sudo journalctl -u cryptix -n 50 --no-pager | grep -i error
# Expected: Few or no errors

# 7. Bot trading
tail -f ~/cryptix/logs/error_log.csv
# Expected: See recent trades

# 8. Dashboard accessible
# Open browser: http://YOUR_IP
# Expected: CRYPTIX dashboard loads
```

---

## 📞 EMERGENCY RECOVERY

### Complete Bot Failure - Nuclear Option

```bash
# 1. Stop everything
sudo systemctl stop cryptix
sudo systemctl stop nginx

# 2. Backup everything
cp -r ~/cryptix ~/cryptix.backup

# 3. Recreate venv
cd ~/cryptix
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Restart
sudo systemctl start cryptix
sudo systemctl start nginx
sudo systemctl status cryptix

# 5. Monitor
sudo journalctl -u cryptix -f
```

### Reset to Clean State

```bash
# If all else fails, terminate and restart instance
# 1. Go to AWS Console > EC2 > Instances
# 2. Select cryptix-bot instance
# 3. Instance State > Terminate
# 4. Create NEW instance with same settings
# 5. Follow setup from AWS_EC2_MIGRATION_GUIDE.md
```

---

## 📊 Quick Diagnostic Commands

Keep these handy for troubleshooting:

```bash
# Overall health check
echo "=== Service ===" && sudo systemctl status cryptix && \
echo "=== Port 5000 ===" && sudo lsof -i :5000 && \
echo "=== Memory ===" && free -h && \
echo "=== Disk ===" && df -h && \
echo "=== Recent errors ===" && sudo journalctl -u cryptix -n 20 --no-pager

# Or create a script
cat > ~/health-check.sh << 'EOF'
#!/bin/bash
echo "=== CRYPTIX Health Check ==="
echo "Service Status:"
sudo systemctl status cryptix | grep -E "Active|Running"
echo ""
echo "Memory Usage:"
free -h | grep -E "Mem:|Swap:"
echo ""
echo "Disk Usage:"
df -h | grep /dev/xvda
echo ""
echo "Python Process:"
ps aux | grep render_launcher | grep -v grep
echo ""
echo "Recent Errors:"
sudo journalctl -u cryptix -n 10 --no-pager | tail -5
EOF

chmod +x ~/health-check.sh
~/health-check.sh
```

---

**Good luck troubleshooting! If issues persist, check:**
1. AWS CloudWatch metrics (CPU, Memory, Network)
2. Security Group rules (may have been modified)
3. VPC routing (unlikely but check)
4. Instance system logs (EC2 Dashboard > Instance > System Log)
