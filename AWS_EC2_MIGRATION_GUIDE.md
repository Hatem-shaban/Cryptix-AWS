# AWS EC2 Migration Guide for CRYPTIX-ML

## Overview
This guide walks you through migrating your CRYPTIX-ML trading bot from Oracle Cloud to **AWS EC2 Free Tier**. The Free Tier gives you 12 months of:
- ✅ **t2.micro** EC2 instance (1 vCPU, 1GB RAM)
- ✅ **30GB** EBS storage (SSD)
- ✅ **1GB** data transfer/month
- ✅ **Free tier eligible** from day 1 of your AWS account

---

## Part 1: AWS Account & EC2 Setup

### Step 1: Access AWS Console
1. Go to **https://console.aws.amazon.com**
2. Sign in with your Amazon account (the one you created)
3. Make sure you're in the **Region** closest to you (e.g., Ireland, Frankfurt, Singapore)

### Step 2: Create EC2 Instance

**Navigate to EC2:**
1. Search for "EC2" in the search bar
2. Click on "EC2" under the Services section
3. Click **"Launch Instances"** button

**Configure Instance:**

**Step 2.1 - Name and OS:**
- **Name**: `cryptix-bot`
- **OS Image**: Ubuntu Server 22.04 LTS (marked as Free Tier Eligible)

**Step 2.2 - Instance Type:**
- **Type**: `t2.micro` (1 vCPU, 1 GiB Memory) - Free Tier Eligible
- **Verify**: You should see a green "Free tier eligible" badge

**Step 2.3 - Key Pair Login:**
1. Click **"Create new key pair"**
2. **Name**: `cryptix-ec2`
3. **Type**: RSA
4. **Format**: `.pem` (for SSH on Mac/Linux) or `.ppk` (for PuTTY on Windows)
5. Click **"Create key pair"** - this downloads to your computer
6. **Store this file safely** - you'll need it to connect!

**Step 2.4 - Network Settings:**
1. Click **"Create security group"**
2. **Name**: `cryptix-sg`
3. **Description**: Security group for CRYPTIX trading bot
4. **Inbound Rules** - Add the following:

| Type | Protocol | Port | Source |
|------|----------|------|--------|
| SSH | TCP | 22 | 0.0.0.0/0 |
| HTTP | TCP | 80 | 0.0.0.0/0 |
| Custom TCP | TCP | 5000 | 0.0.0.0/0 |

5. **Outbound Rules**: Keep default (allow all)

**Step 2.5 - Storage:**
- **Volume Type**: gp3 (General Purpose SSD)
- **Size**: 30 GB (Free tier eligible)
- **Delete on Termination**: ✓ Checked

**Step 2.6 - Advanced (Optional):**
- Scroll down to **"Monitoring"**
- Enable **"Detailed CloudWatch monitoring"** (small cost, helps debug issues)

**Step 2.7 - Review and Launch:**
1. Review all settings
2. Click **"Launch Instance"**
3. Your instance will start in ~30 seconds

### Step 3: Get Your Instance Public IP

1. Go to **EC2 Dashboard > Instances**
2. Click on your `cryptix-bot` instance
3. Note the **Public IPv4 address** (e.g., `51.20.60.192`)
4. Wait for **Instance State** to show "running" and **Status Checks** to be "2/2 passed"

---

## Part 2: Connect to Your EC2 Instance

### On Windows (PowerShell):

**A. Set Key Permissions:**
```powershell
# Navigate to where you downloaded cryptix-ec2.pem
cd $env:USERPROFILE\Downloads

# Set restrictive permissions (required for SSH)
icacls "cryptix-ec2.pem" /grant:r "$env:USERNAME`:`(F`)"
icacls "cryptix-ec2.pem" /inheritance:r
```

**B. Connect via SSH:**
```powershell
# Replace 51.20.60.192 with your actual instance IP
ssh -i "cryptix-ec2.pem" ubuntu@51.20.60.192
```

You should see:
```
The authenticity of host '51.20.60.192' can't be established...
Are you sure you want to continue connecting (yes/no/fingerprint)?
```
Type `yes` and press Enter.

**C. If SSH Fails:**
- Error: "Connection timed out" → Wait 5 minutes, instance may still be starting
- Error: "Permission denied" → Check key permissions with `icacls` command above
- Error: "No such file" → Verify the path to `cryptix-ec2.pem` is correct

---

## Part 3: Server Setup

Once connected via SSH, run these commands on your EC2 instance:

### Step 1: Update System
```bash
sudo apt update
sudo apt upgrade -y
```

### Step 2: Install Dependencies
```bash
sudo apt install -y python3 python3-pip python3-venv git curl wget nginx
```

### Step 3: Clone Your Repository
```bash
git clone https://github.com/Hatem-shaban/Cryptix-AWS.git ~/Cryptix-AWS
cd ~/Cryptix-AWS
```

### Step 4: Create Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 5: Install Python Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you get wheel compilation errors, try this alternative:
```bash
pip install -r requirements.txt --no-build-isolation
```

### Step 6: Configure Environment
```bash
cp .env .env.backup    # Backup existing config
nano .env              # Edit your configuration
```

**Edit these values in the .env file:**
```dotenv
# Keep your existing values:
TELEGRAM_BOT_TOKEN=8244322664:AAFMhtmip4JiX-qk5Xobzdn9CzejRh00Ti4
TELEGRAM_CHAT_ID=2086996577
API_KEY=xiOuLrTaSvHer9Arr4rkuj742PxEkp5jW4JlaQxPBk1TxOzD1mTVBHUxknhUPAxR
API_SECRET=zWGMloL3OFejQZsE7gxb2reH6EJXO0FLQG3VvL9HU5Tbkvkgnw7UrWLFOgvPbc7d
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRsZnBzbnl6dXJsb2NxY2l5dnF6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDgyOTgyNCwiZXhwIjoyMDc2NDA1ODI0fQ.76IxSOJSnwggAe9lBuXht9c-2MtRRtCm6Y_75BgzksY
SUPABASE_URL=https://dlfpsnyzurlocqciyvqz.supabase.co

# AWS-specific addition:
RENDER_DEPLOYMENT=false  # This disables Render-specific optimizations
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### Step 7: Test the Bot
```bash
# Test configuration
python render_launcher.py &
sleep 5
curl http://localhost:5000/health
```

You should see a JSON response with health status.

---

## Part 4: Setup Systemd Service (Auto-Start)

### Step 1: Create Service File
```bash
sudo nano /etc/systemd/system/cryptix.service
```

### Step 2: Paste Configuration
```ini
[Unit]
Description=CRYPTIX Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/Cryptix-AWS
Environment="PATH=/home/ubuntu/Cryptix-AWS/venv/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/home/ubuntu/Cryptix-AWS/.env
ExecStart=/home/ubuntu/Cryptix-AWS/venv/bin/python render_launcher.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### Step 3: Enable and Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable cryptix
sudo systemctl start cryptix
sudo systemctl status cryptix
```

You should see:
```
● cryptix.service - CRYPTIX Trading Bot
     Loaded: loaded (...; enabled; vendor preset: enabled)
     Active: active (running)...
```

---

## Part 5: Setup Nginx Reverse Proxy

This allows you to access your bot at `http://your-instance-ip` instead of `http://your-instance-ip:5000`

### Step 1: Create Nginx Configuration
```bash
sudo nano /etc/nginx/sites-available/cryptix
```

### Step 2: Paste Configuration
```nginx
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

### Step 3: Enable the Site
```bash
sudo ln -s /etc/nginx/sites-available/cryptix /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default    # Remove default site
sudo nginx -t                               # Test configuration
sudo systemctl restart nginx
```

### Step 4: Access Your Bot
Open your browser and go to: `http://51.20.60.192` 

---

## Part 6: Monitor and Manage

### View Bot Status
```bash
sudo systemctl status cryptix
sudo journalctl -u cryptix -n 50 --no-pager    # Last 50 log lines
sudo journalctl -u cryptix -f                  # Live logs
```

### View Application Logs
```bash
tail -f ~/Cryptix-AWS/logs/error_log.csv
ls -lh ~/Cryptix-AWS/logs/
```

### Restart Bot
```bash
sudo systemctl restart cryptix
sleep 3
sudo systemctl status cryptix
```

### Stop/Start Bot
```bash
sudo systemctl stop cryptix     # Stop
sudo systemctl start cryptix    # Start
```

### Update Bot Code
```bash
cd ~/Cryptix-AWS
git pull
sudo systemctl restart cryptix
```

---

## Part 7: Monitoring & Optimization

### Check Memory Usage
```bash
free -h
top    # Press 'q' to exit
```

For AWS t2.micro with 1GB RAM:
- Ideal usage: < 700MB (leaves buffer for system)
- Warning: > 850MB (risk of OOM kill)

### Check Disk Usage
```bash
df -h
du -sh ~/cryptix/logs/
```

If logs get too large:
```bash
find ~/Cryptix-AWS/logs -name "*.csv" -mtime +30 -delete    # Delete logs > 30 days old
gzip ~/Cryptix-AWS/logs/error_log.csv                       # Compress current log
```

### Check CPU Usage
```bash
uptime
ps aux | grep python
```

### Enable CloudWatch Monitoring (Optional)
For better monitoring through AWS dashboard:
1. Go to **EC2 Dashboard > Instances**
2. Select your instance
3. Go to **Monitoring** tab
4. View CPU, Network metrics over time

---

## Part 8: Security Considerations

### 1. Update SSH Security Group (Recommended)
Instead of allowing SSH from anywhere (0.0.0.0/0), restrict to your IP:

1. Go to **EC2 > Security Groups > cryptix-sg**
2. Edit **Inbound Rules**
3. Find SSH rule (port 22)
4. Change Source from `0.0.0.0/0` to your public IP (find at https://whatismyipaddress.com)
5. Add `/32` to the end (e.g., `203.45.123.78/32`)
6. Save

### 2. Keep Private Keys Safe
```bash
# On your local machine, restrict key permissions
chmod 600 cryptix-ec2.pem
```

### 3. Rotate API Keys Periodically
1. Go to Binance API management
2. Generate new API key every 3-6 months
3. Update .env on your instance
4. Restart the bot

### 4. Regular Backups
Periodically download your trades and configuration:
```bash
# From your local machine:
scp -i "cryptix-ec2.pem" -r ubuntu@51.20.60.192:~/cryptix/logs ./backup-$(date +%Y%m%d)
```

---

## Part 9: Troubleshooting

### Bot Won't Start
**Error**: `cryptix.service failed to start`

**Solution**:
```bash
# Check detailed error
sudo journalctl -u cryptix -n 100 --no-pager
# Check if port 5000 is in use
sudo lsof -i :5000
# Check Python environment
~/Cryptix-AWS/venv/bin/python --version
```

### Connection to Binance API Fails
**Error**: `Failed to connect to Binance API`

**Solution**:
```bash
# Verify API credentials in .env
cat ~/.env | grep API_KEY
# Test API connectivity
curl https://api.binance.com/api/v3/time
# Restart with debug logs
sudo journalctl -u cryptix -f
```

### Out of Memory
**Error**: `Killed (OOM)`

**Solution** (for t2.micro):
1. Increase swap space:
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

2. Or reduce historical data in config:
```bash
nano ~/Cryptix-AWS/config.py
# Reduce LOOKBACK_PERIOD or BATCH_SIZE
```

### Nginx 404 Error
**Error**: `404 Not Found`

**Solution**:
```bash
# Check if Flask is running
ps aux | grep python
# Check Nginx logs
sudo tail -f /var/log/nginx/error.log
# Restart both services
sudo systemctl restart cryptix nginx
```

---

## Part 10: Cleanup (Stopping Costs)

**Important**: To avoid unnecessary charges after the 12-month free tier:

### Temporary Stop (pause costs):
```bash
# From AWS Console:
# 1. Go to EC2 > Instances
# 2. Select cryptix-bot instance
# 3. Click Instance State > Stop
# 4. Data preserved, costs pause
```

### Permanent Delete (delete all):
```bash
# From AWS Console:
# 1. Go to EC2 > Instances
# 2. Select cryptix-bot instance
# 3. Click Instance State > Terminate
# 4. All data deleted, instance removed
```

---

## Part 11: Migration Checklist

- [ ] Created AWS account and verified Free Tier eligibility
- [ ] Launched t2.micro EC2 instance with Ubuntu 22.04
- [ ] Created security group with SSH, HTTP, and port 5000 access
- [ ] Generated and downloaded cryptix-ec2.pem key pair
- [ ] Connected to instance via SSH
- [ ] Installed dependencies and created virtual environment
- [ ] Cloned CRYPTIX-ML repository
- [ ] Configured .env with API credentials
- [ ] Tested bot with health check
- [ ] Created systemd service for auto-start
- [ ] Configured Nginx reverse proxy
- [ ] Accessed bot dashboard at http://instance-ip
- [ ] Verified bot is trading (check logs)
- [ ] Restricted SSH security group to your IP
- [ ] Set up log backups
- [ ] Monitored resource usage

---

## Migration Complete! 🎉

Your CRYPTIX trading bot is now running on AWS EC2 Free Tier with:
- ✅ **Auto-start** on instance reboot
- ✅ **Nginx reverse proxy** for easy web access
- ✅ **Persistent logs** and trade history
- ✅ **Supabase integration** for position tracking
- ✅ **Telegram notifications** still working
- ✅ **12 months free** (then ~$7-10/month if continued)

### Next Steps:
1. Monitor the bot for 24-48 hours
2. Verify trades are executing correctly
3. Check Telegram notifications are coming through
4. Set up periodic log backups
5. After 12 months: Upgrade to larger instance or migrate to Render

---

## Quick Commands Reference

```bash
# Connect to instance
ssh -i "$env:USERPROFILE\Downloads\cryptix-ec2.pem" ubuntu@51.20.60.192

# View bot status
sudo systemctl status cryptix

# View live logs
sudo journalctl -u cryptix -f

# Restart bot
sudo systemctl restart cryptix

# Update code
cd ~/cryptix && git pull && sudo systemctl restart cryptix

# Check resources
free -h && df -h

# View Flask dashboard
# Open browser to: http://51.20.60.192
```

---

## Comparison: Oracle Cloud vs AWS EC2 vs Render

| Feature | Oracle Cloud | AWS EC2 Free | Render |
|---------|--------------|--------------|--------|
| **Cost (Free Tier)** | 3 months | 12 months | Pay-as-you-go |
| **RAM** | 1-2GB | 1GB (t2.micro) | 512MB-2GB |
| **CPU** | 1 core | 1 vCPU | Shared |
| **Storage** | 100GB | 30GB | 1GB system |
| **Setup Complexity** | Medium | Low | Very Low |
| **Uptime** | Excellent | Excellent | Excellent |
| **Scaling** | Easy | Easy | Easy |

**Recommendation**: AWS EC2 Free Tier is ideal for you because:
1. **12 months free** (vs Oracle's 3 months)
2. **Simple setup** (just like Oracle)
3. **No payment required** during free tier
4. **Easy to scale** after free tier ends

---

## Support & References

- **AWS EC2 Documentation**: https://docs.aws.amazon.com/ec2/
- **AWS Free Tier Details**: https://aws.amazon.com/free/
- **Ubuntu on AWS**: https://ubuntu.com/aws/
- **Systemd Tutorial**: https://www.digitalocean.com/community/tutorials/understanding-systemd-units-and-unit-files

Good luck with your migration! 🚀
