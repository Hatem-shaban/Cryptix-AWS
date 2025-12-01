# Oracle Cloud VM Setup Guide for CRYPTIX-ML

## Troubleshooting SSH Connection Issues

If you get "Connection timed out" when trying to SSH:

### **Issue: Network Connectivity Problem**

The most common causes are:
1. **Subnet's route table not configured** - Traffic can't reach the Internet Gateway
2. **Security group blocking traffic** - Though we added rules, they may not be applied
3. **Instance still booting** - Wait 5-10 minutes after creation

### **Solution Steps:**

**Step 1: Verify Internet Gateway is Attached**
1. Go to **Networking > Virtual Cloud Networks > cryptix-vcn**
2. Click **Internet Gateways** (left sidebar)
3. You should see an Internet Gateway in "AVAILABLE" state
4. If not present, click "Create Internet Gateway" and name it "cryptix-igw"

**Step 2: Verify Route Table Configuration - SIMPLIFIED APPROACH**

Since the subnet is correctly associated with the Default Route Table, let's try a different approach:

1. In VCN details, click **Route Tables** (left sidebar)
2. Click on the **Default Route Table for cryptix-vcn**
3. Look for existing routes - if you see a route with:
   ```
   Destination: 0.0.0.0/0
   Target: [IGW Name]
   State: Active
   ```
   Then the route is already configured ✓

4. If NO route exists, click **"Add Route Rule"** and select:
   - **Target Type**: Drop down and select **"Internet Gateway"**
   - **Destination CIDR Block**: `0.0.0.0/0`
   - **Target Internet Gateway**: `cryptix-igw`
   - Click **"Add Route Rules"**

**If you get an API error about "private IP":**
- This is a known Oracle Cloud issue. Try this workaround:
- Cancel the dialog
- Go to **VCN > Actions > Enable "Skip Source/Destination Check"** if available
- Then try adding the route again

**Alternative: Use Oracle Cloud Console UI Wizard**
If manual route addition keeps failing:
1. Delete the current VCN
2. Go to **Virtual Cloud Networks > Start VCN Wizard**
3. Select **"Create VCN with Internet Connectivity"** (this auto-configures routes)
4. Complete the wizard
5. This will automatically set up all routing correctly

**Step 3: Verify Security List Rules Again**
1. Go to **Security Lists > Default Security List**
2. Ensure you have ALL these rules:
   - SSH (port 22) - Source: 0.0.0.0/0
   - HTTP (port 80) - Source: 0.0.0.0/0  
   - Flask (port 5000) - Source: 0.0.0.0/0

**Step 4: Reboot the Instance**
1. Go to **Instances > instance-20251111-1523**
2. Click "Reboot" button
3. Wait 2-3 minutes for reboot to complete
4. Try SSH again

**Step 5: Try SSH Again**
```powershell
ssh -i "$env:USERPROFILE\.ssh\cryptix_ovm" opc@150.8.17.63
```

---

## Initial VM Setup

1. Generate and Configure SSH Keys:

a. Generate SSH Key Pair (Windows PowerShell):
```powershell
# Create .ssh directory if it doesn't exist
mkdir -force "$env:USERPROFILE\.ssh"

# Generate SSH key pair
ssh-keygen -t rsa -b 2048 -f "$env:USERPROFILE\.ssh\cryptix_ovm"
```
When prompted:
- Press Enter to use the default location
- Enter a passphrase (recommended) or press Enter twice for no passphrase

b. During Instance Creation:
- In the "Add SSH keys" section, choose "Paste public key"
- Open the public key file:
```powershell
Get-Content "$env:USERPROFILE\.ssh\cryptix_ovm.pub"
```
- Copy the entire content and paste it into the SSH key field

c. After Instance Creation, SSH into your Oracle VM:
```powershell
ssh -i "$env:USERPROFILE\.ssh\cryptix_ovm" opc@<your-vm-ip>
```
Note: Replace <your-vm-ip> with the public IP address shown in the instance details

2. Update system and install dependencies:
```bash
sudo dnf update -y
sudo dnf groupinstall "Development Tools" -y
sudo dnf install python39 python39-devel python39-pip git nginx -y
```

3. Configure Network Access:

a. Create Virtual Cloud Network (VCN) using the Wizard:
   - Go to Oracle Cloud Console > Networking > Virtual Cloud Networks
   - Click "Start VCN Wizard"
   - Select "Create VCN with Internet Connectivity"
   - Click "Start VCN Wizard"
   - Fill in the following details:
     ```
     Basic Information:
     VCN Name: cryptix-vcn
     Compartment: (your compartment)
     VCN CIDR Block: 10.0.0.0/16
     Public Subnet CIDR Block: 10.0.0.0/24
     Private Subnet CIDR Block: 10.0.1.0/24
     ```
   - Click "Next" and review the configuration
   - Click "Create" to create the VCN and subnets

b. During Compute Instance Creation:
   - In the "Placement" section:
     * Availability domain: Try each AD in sequence (AD-1, AD-2, then AD-3)
     * Fault domain: Select "Let Oracle choose the best fault domain"
     * Capacity type: "On-demand capacity" (default)

   - In the "Image and Shape" section:
     * Image: Oracle Linux 9
     * Shape: Alternative options if VM.Standard.A1.Flex is out of capacity:
       
       **Option 1 - VM.Standard.E2.1.Micro (Always Free Eligible)**
       - Most reliable Always Free option
       - 1 OCPU, 1GB RAM (sufficient for trading bot with memory optimization)
       - Usually has better availability
       
       **Option 2 - VM.Standard.E3.Flex**
       - If E2.1.Micro is also unavailable
       - Select: 1 OCPU, 1GB Memory
       - Pay-as-you-go but eligible for free credits

   - In the "Networking" section:
     * Select "Select existing virtual cloud network"
     * Choose "cryptix-vcn" from the dropdown
     * For subnet, select "cryptix-subnet"
     * Ensure "Assign a public IPv4 address" is checked
     * You can safely ignore the IPv6 warning - our application only requires IPv4

## Capacity Issues - Solutions

If VM.Standard.A1.Flex is unavailable in all ADs:

**Option 1: VM.Standard.E3.Flex (Recommended if available)**
1. In "Browse all shapes", look for "VM.Standard.E3.Flex"
2. Select it
3. Set: 1 OCPU, 1GB Memory
4. May incur small charges but eligible for free trial credits
5. Click "Select shape"

**Option 2: Check "Specialty and previous generation" tab**
1. Click on "Specialty and previous generation" in the shapes browser
2. Look for "VM.Standard.E2.1.Micro" or other Always Free eligible shapes
3. These are usually more available
4. Select the shape and click "Select shape"

**Option 3: Change Image to Oracle Linux 8**
1. Go back to "Image and shape" section
2. Change Image from "Oracle Linux 9" to "Oracle Linux 8"
3. Then try E2.1.Micro again (may have better compatibility)
4. If still unavailable, try E3.Flex

**Option 4: Try Different Availability Domain BEFORE changing shape**
1. In "Placement" section, try a different AD (AD-1, AD-2, or AD-3)
2. Each AD may have different availability
3. Then try the original shape again

**Timing Strategy:**
- Try creating instances during off-peak hours (early morning, late night UTC)
- Free tier resources are more available in these periods
- Oracle typically has better availability on weekdays

c. Configure Security List:
   - In your VCN details page, click "Security Lists"
   - Click on the Default Security List
   - Click "Add Ingress Rules" and configure:
     ```
     Source CIDR: 0.0.0.0/0
     IP Protocol: TCP
     Source Port Range: All
     Destination Port Range: 5000
     Description: Flask Web Interface
     ```
   - Add another rule for HTTP:
     ```
     Source CIDR: 0.0.0.0/0
     IP Protocol: TCP
     Destination Port Range: 80
     Description: Nginx Web Server
     ```

b. Configure Local Firewall:
```bash
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --reload
```

Note: Both Oracle Cloud security list and local firewall must allow port 5000 for the Flask application to be accessible externally.

4. Create project directory:
```bash
mkdir -p ~/cryptix
cd ~/cryptix
```

5. Clone repository:
```bash
git clone https://github.com/your-username/CRYPTIX-ML-02.git .
```

## Python Environment Setup

1. Create and activate virtual environment:
```bash
python3.9 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy environment file:
```bash
cp .env.example .env
# Edit .env with your credentials
nano .env
```

## Nginx Setup for Dashboard

1. Create Nginx configuration:
```bash
sudo nano /etc/nginx/conf.d/cryptix.conf
```

2. Add configuration:
```nginx
server {
    listen 80;
    server_name your-vm-ip;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /logs {
        alias /home/opc/cryptix/logs;
        autoindex on;
    }
}
```

3. Test and restart Nginx:
```bash
sudo nginx -t
sudo systemctl restart nginx
```

## Systemd Service Setup

1. Create service file:
```bash
sudo nano /etc/systemd/system/cryptix.service
```

2. Add service configuration:
```ini
[Unit]
Description=CRYPTIX Trading Bot
After=network.target

[Service]
Type=simple
User=opc
WorkingDirectory=/home/opc/cryptix
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=/home/opc/cryptix/.env
ExecStart=/home/opc/cryptix/venv/bin/python web_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. Enable and start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable cryptix
sudo systemctl start cryptix
```

## Log Access Setup

1. Configure log directory permissions:
```bash
chmod 755 ~/cryptix/logs
```

2. View logs via:
- Web dashboard: http://your-vm-ip/logs
- Command line: `journalctl -u cryptix -f`
- Direct log files: `tail -f ~/cryptix/logs/*.csv`

## Health Checks

1. Check service status:
```bash
sudo systemctl status cryptix
```

2. Monitor application logs:
```bash
tail -f ~/cryptix/logs/error_log.csv
```

3. Access dashboard:
http://your-vm-ip

## Maintenance Tasks

1. Update application:
```bash
cd ~/cryptix
git pull
sudo systemctl restart cryptix
```

2. Rotate logs:
```bash
find ~/cryptix/logs -name "*.csv" -mtime +30 -exec gzip {} \;
```

## Emergency Recovery

If the bot crashes or needs restart:
```bash
sudo systemctl restart cryptix
```

To check recent errors:
```bash
journalctl -u cryptix -n 100 --no-pager
```