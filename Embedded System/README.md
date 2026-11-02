# Bat Activity Monitoring System for Raspberry Pi Zero 2 W

**Real-time bat vocalization monitoring with autonomous email reporting**

## Overview

This system transforms a Raspberry Pi Zero 2 W into a standalone bat activity monitor that:
- ✅ Captures ultrasonic audio from bats via I2S microphone
- ✅ Classifies vocalizations in real-time using TensorFlow Lite
- ✅ Tracks activity patterns by species and behavior
- ✅ Automatically starts/stops based on configured schedule
- ✅ Sends activity reports and food consumption predictions via email
- ✅ **Never stores audio files** - processes everything in real-time

## Hardware Requirements

### Core Components
- **Raspberry Pi Zero 2 W** - Main processor
- **8GB+ microSD card** - Operating system and software
- **5V Power Supply** - Official Raspberry Pi power adapter recommended

### Sensors & Peripherals
- **INMP441 I2S MEMS Microphone** - Digital ultrasonic audio capture
- **DS3231 RTC Module** - Real-time clock for accurate scheduling
- **2.0" TFT LCD Display with EC11 Rotary Encoder (ST7789)** - Full-color touchscreen interface with integrated encoder for menu navigation

### Optional
- USB WiFi dongle (if built-in WiFi signal is weak)
- Battery backup for DS3231 (CR2032 coin cell)

## Wiring Diagram

### I2S Microphone (INMP441)
```
INMP441          Raspberry Pi Zero 2 W
-------          ---------------------
VDD       -->    3.3V (Pin 1)
GND       -->    GND (Pin 6)
SCK       -->    GPIO 18 / BCM 18 (Pin 12)
WS        -->    GPIO 19 / BCM 19 (Pin 35)
SD        -->    GPIO 21 / BCM 21 (Pin 40)
L/R       -->    GND (for left channel)
```

### RTC Module (DS3231)
```
DS3231           Raspberry Pi Zero 2 W
-------          ---------------------
VCC       -->    3.3V (Pin 1)   ← not 5V: see the battery note below
GND       -->    GND (Pin 6)
SDA       -->    GPIO 2 / SDA (Pin 3)
SCL       -->    GPIO 3 / SCL (Pin 5)
```

Power the RTC from **3.3 V**:
- **Safe I2C levels:** most DS3231 boards pull SDA/SCL up to their VCC, and the Pi's pins are 3.3 V only.
- **Battery:** the common DS3231 board with a CR2032 holder on the back ("ZS-042") has a charging circuit meant for a *rechargeable* LIR2032. On 5 V it pushes about 4.3 V into a normal, non-rechargeable CR2032, which can make it swell or leak. On 3.3 V it hardly charges at all.

A CR2032 doesn't need charging anyway: the clock draws only a few microamps from it, so it lasts for years.

### 2.0" TFT LCD Display with EC11 Rotary Encoder (SPI)
```
TFT Display      Raspberry Pi Zero 2 W
-----------      ---------------------
VCC       -->    3.3V (Pin 17)
GND       -->    GND (Pin 20)
SCL       -->    GPIO 11 / SPI0 SCLK (Pin 23)
SDA       -->    GPIO 10 / SPI0 MOSI (Pin 19)
RES       -->    GPIO 25 (Pin 22)
DC        -->    GPIO 24 (Pin 18)
CS        -->    GPIO 8 / SPI0 CE0 (Pin 24)
BLK       -->    GPIO 23 (Pin 16) - Backlight control

EC11 Encoder     Raspberry Pi Zero 2 W
------------     ---------------------
CLK       -->    GPIO 17 (Pin 11)
DT        -->    GPIO 27 (Pin 13)
SW        -->    GPIO 22 (Pin 15)
VCC       -->    3.3V (Pin 1) - shared
GND       -->    GND (Pin 6) - shared
```

**Display Features:**
- Resolution: 240x320 pixels
- Full color (65K colors)
- ST7789 driver chip
- Integrated EC11 rotary encoder
- SPI interface for fast updates

## Software Installation

### 1. Prepare Raspberry Pi

```bash
# Flash Raspberry Pi OS Lite (64-bit) to microSD card
# Boot up and connect to WiFi

# Update system
sudo apt update
sudo apt upgrade -y
```

### 2. Download System Files

```bash
# Create project directory
mkdir ~/bat-monitor
cd ~/bat-monitor

# Copy all Python files and models to this directory
# - main.py
# - config_manager.py
# - audio_capture.py
# - feature_extraction.py
# - model_inference.py
# - activity_tracker.py
# - email_sender.py
# - rtc_scheduler.py
# - ui_controller.py
# - batmon.py (terminal control panel), web_ui.py, web_ui.html, wifi_manager.py
# - bat_model.tflite            (AI model, retrained Oct 2026)
# - label_encoder.pkl
# - 12_29_both_species.tflite    (original model, kept as a fallback)
# - install.sh
# - bat_monitor.service
```

### 3. Run Installation Script

```bash
cd ~/bat-monitor
chmod +x install.sh
sudo ./install.sh
```

The installation script will:
- Install system dependencies
- Enable I2C and I2S interfaces
- Create Python virtual environment
- Install all Python packages
- Create systemd service
- Generate hardware test script

### 4. Reboot

```bash
sudo reboot
```

### 5. Test Hardware

After reboot:

```bash
cd ~/bat-monitor
source venv/bin/activate
python test_hardware.py
```

Expected output:
```
=== I2C Devices ===
Found devices: 0x3c, 0x68
  ✓ DS3231 RTC detected (0x68)
  ✓ SSD1306 OLED detected (0x3C)

=== Audio Devices ===
Available devices:
  [1] i2s (input)

=== GPIO ===
  ✓ GPIO initialized

=== AI Model ===
  ✓ Model loaded (7 classes)

✓ All tests passed! System ready.
```

## Configuration

### Control Panel (in the terminal, over SSH)

SSH into the Pi and the control panel opens automatically:

```bash
ssh pi@raspberrypi.local
```

It's a full-screen terminal app (`batmon`). Use the keyboard, or the mouse in
most terminals:

| key | does |
|---|---|
| `1`–`8` | Dashboard, Schedule, Email, Detection, WiFi, Clock, System, Logs |
| `s` | start / stop monitoring (stopping asks first, then emails the report) |
| `ctrl+s` | save the current page |
| `Tab` / `Esc` | move between fields / leave a text field |
| `q` | quit to the normal shell |

From there you can:

- **Dashboard:** see tonight's call counts, the live microphone level, the food prediction, recent detections and system health
- **Schedule:** edit the start and stop times
- **Email:** change the account and recipients, the food-prediction trendline, and send a test email
- **Detection:** tune detection sensitivity while watching the live level
- **WiFi:** join a network
- **Clock:** set the clock
- **System:** restart the monitor service
- **Logs:** read the logs

Changes apply immediately. If the monitor service isn't running, the panel
says so and still lets you edit settings. They're used when the service
starts (there's a **Start service** button).

Type `batmon` to reopen it. `batmon --autostart off` stops it opening on
login (`on` turns it back on).

**Calibrating detection:** the AI model gives every sound a label, even
silence, so only sounds louder than the *energy threshold* are classified.
On the Detection page (`4`), watch the meter during a quiet period and set the
threshold just above the background level.

### Email Settings

Edit `config.json`:

```json
{
  "email": {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your-email@gmail.com",
    "sender_password": "your-app-password",
    "recipients": ["keeper1@zoo.com", "keeper2@zoo.com"]
  }
}
```

**For Gmail:**
1. Enable 2-Factor Authentication
2. Generate App Password: https://myaccount.google.com/apppasswords
3. Use the app password (not your regular password)

### Time and time zone

Every time the monitor shows or uses is in one time zone: the recording schedule, the device screen, the control panel, logs and reports. The default is Pacific (the Oregon Zoo). Change it on the device under **Set clock → Time zone**, or in the control panel under **Clock** (key `6`). It applies immediately.

The DS3231 clock chip keeps UTC, so daylight-saving changes never put it out. With internet, the chip is kept on internet time (checked hourly). Without internet, the Pi sets its own clock from the chip at startup. If the time is wrong, use **Set clock → Date & time** on the device, or **Clock** in the control panel.

### WiFi (and getting back in if it's wrong)

**Joining a network:** on the device, open **WiFi → Choose network…**, pick yours from the list and type only the password. The top of the WiFi menu (and **Status**) shows which network the Pi is on and its IP address, which is the address to SSH to.

**Setup hotspot:** if the Pi can't join any network about 90 seconds after it starts (wrong password, new place, router off), it creates its own WiFi network so you're never locked out:

1. Connect your laptop or phone to the WiFi network **BatPi-Setup** (password `batsetup123`).
2. SSH in with `ssh <your-pi-user>@10.42.0.1`. The control panel opens; use **WiFi** (key `5`) to pick a network. Or use the device's WiFi menu.
3. The hotspot turns off by itself once the Pi joins a network. Every 5 minutes it also briefly retries the saved networks, so after a power cut it reconnects as soon as the router is back. Nobody connected to the hotspot is interrupted.

Change the hotspot password in `config.json` (`wifi.hotspot_password`), or turn the hotspot off with `wifi.setup_hotspot: false`.

**Hotel, school and guest WiFi** usually block devices from reaching each other and need a sign-in page in a browser. The Pi can't use those: use a phone hotspot or a normal home or zoo network.

**No internet at night?** Reports are never lost. If one can't be emailed, it's kept in `reports/outbox` and sent automatically once the Pi is back online, even after a restart.

### Using the Device Screen and Knob

The knob works the same way on every screen:

| do this | to |
|---|---|
| **turn** | move through a list, or change a number (spin faster for bigger steps) |
| **click** | select / confirm / go to the next number |
| **hold** (about 1 second) | go back, or cancel without saving |

The bottom line of the screen always shows what the knob does there.

**Main menu:**
- **Start / Stop monitoring:** stopping asks you to confirm first. Tonight's report is then emailed.
- **Status:** shows the time, schedule, recipients and trendline.
- **Schedule:** sets the start time, stop time, and schedule on/off. Times are edited as hours and then minutes.
- **Set clock:** sets the year, month, day, hour and minute.
- **Email:** sets the sender, password and recipients (add, edit or remove), and sends a test email.
- **WiFi:** sets the network name and password, then connects.
- **Food trendline:** sets the slope and intercept. Spin fast to change by whole numbers.
- **Screen & knob:** turns the screen 180°, reverses the knob direction, and sets how long before the screen turns off.

Text is entered with an on-screen keyboard. Turn to move between keys, click to type, and pick OK to save. `.com` types the whole ending, and SYM switches to symbols.

**Good to know:**
- **Screen sleep:** the screen turns off after 1 minute without input, which keeps light away from the bats. Your first touch only wakes it up.
- **Knob type:** detected automatically, so one click is always one step.
- **Wrong direction:** if turning moves the wrong way, use **Screen & knob → Reverse knob**.
- **Screen orientation:** if the screen appears upside down after reassembly, use **Screen & knob → Turn screen 180°**.

## Running the System

### Manual Start

```bash
cd ~/bat-monitor
source venv/bin/activate
python main.py
```

### Auto-Start on Boot

```bash
# Enable service
sudo systemctl enable bat-monitor.service

# Start service
sudo systemctl start bat-monitor.service

# Check status
sudo systemctl status bat-monitor.service

# View logs
sudo journalctl -u bat-monitor.service -f
```

### Stop Service

```bash
sudo systemctl stop bat-monitor.service
```

## Operation

### Typical Workflow

1. **Setup**: Connect hardware, install software, configure email
2. **Configure Schedule**: Use rotary encoder to set start time (e.g., 18:00) and stop time (e.g., 06:00)
3. **Deploy**: Place system in bat enclosure, plug in power
4. **Automatic Operation**: 
   - At 18:00, system starts capturing audio
   - Real-time classification happens continuously
   - Audio is NOT stored - only activity data
   - At 06:00, system stops and sends email report
5. **Receive Report**: Email contains activity graphs, behavior breakdown, and food prediction

### What Gets Sent via Email

The email report includes:

1. **Activity Timeline Graph** - Hourly activity by species
2. **Behavior Breakdown** - Pie charts showing behavior distribution
3. **Food Consumption Prediction** - Based on "Want_Food" call frequency
4. **Statistics Table**:
   - Total vocalizations
   - Species counts
   - Behavior counts per species
   - Recording duration

### Data Storage

**What is stored:**
- Activity counts (hourly bins)
- Classification results (species, behavior, timestamp)
- Activity graphs (PNG images, deleted after email sent)
- Log files (`bat_monitor.log`)

**What is NOT stored:**
- Raw audio files
- Audio spectrograms
- Individual audio segments

Total storage per day: < 1 MB

## Maintenance

### View Logs

```bash
# System service logs
sudo journalctl -u bat-monitor.service -n 100

# Application log file
tail -f ~/bat-monitor/bat_monitor.log
```

### Update Configuration

```bash
cd ~/bat-monitor
nano config.json
# Make changes
# Restart service
sudo systemctl restart bat-monitor.service
```

### Backup Configuration

```bash
cp ~/bat-monitor/config.json ~/config_backup.json
```

## Troubleshooting

### I2C Devices Not Detected

```bash
# Check I2C is enabled
sudo raspi-config
# Interface Options -> I2C -> Enable

# Scan I2C bus
sudo i2cdetect -y 1

# Expected output shows 0x68 (RTC)
# Note: TFT display uses SPI, not I2C
```

### TFT Display Not Working

```bash
# Check SPI is enabled
sudo raspi-config
# Interface Options -> SPI -> Enable

# Verify SPI devices exist
ls -l /dev/spidev*
# Should show: /dev/spidev0.0 and /dev/spidev0.1

# Check wiring:
# - VCC to 3.3V (NOT 5V!)
# - All SPI pins connected (MOSI, SCLK, CS, DC, RST)
# - Backlight wire to GPIO 23
```

### Audio Not Capturing

```bash
# Check I2S overlay
# (Raspberry Pi OS Bookworm and later: /boot/firmware/config.txt)
grep -E "dtparam=i2s=on|googlevoicehat-soundcard" /boot/firmware/config.txt

# If not present, add them:
echo "dtparam=i2s=on" | sudo tee -a /boot/firmware/config.txt
echo "dtoverlay=googlevoicehat-soundcard" | sudo tee -a /boot/firmware/config.txt
sudo reboot

# Check the microphone appears as a capture device
arecord -l

# Test audio devices
source ~/bat-monitor/venv/bin/activate
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Email Not Sending

```bash
# Test email manually
cd ~/bat-monitor
source venv/bin/activate
python -c "from email_sender import *; from config_manager import *; \
config = ConfigManager(); \
sender = EmailSender(config.get_email_config()); \
print(sender.test_connection())"
```

Check:
- SMTP credentials correct
- App password (not regular password for Gmail)
- WiFi connected
- Firewall not blocking SMTP port 587

### System Clock Wrong

```bash
# Check RTC time
sudo hwclock -r

# Set RTC to system time
sudo hwclock -w

# Set system time from RTC
sudo hwclock -s

# Or set manually
sudo date -s "2024-04-05 14:30:00"
sudo hwclock -w
```

### High CPU/Memory Usage

Check service status:
```bash
sudo systemctl status bat-monitor.service
```

Monitor resources:
```bash
htop
```

The system should use:
- CPU: ~50-70% during monitoring
- Memory: 200-350 MB

### Model Inference Too Slow

If classification can't keep up with real-time audio:

1. Reduce chunk duration in `config.json`:
   ```json
   "audio": {
     "chunk_duration": 5.0
   }
   ```

2. Lower sample rate (if model allows):
   ```json
   "audio": {
     "sample_rate": 32000
   }
   ```

## Advanced Configuration

### Custom Trendline for Food Prediction

Edit `config.json`:

```json
{
  "email": {
    "trendline_slope": 6.4899,
    "trendline_intercept": 40.2899
  }
}
```

### Adjust Detection Sensitivity

Edit `config.json`:

```json
{
  "audio": {
    "energy_threshold": 0.001,
    "min_vocalization_duration": 0.5,
    "max_vocalization_duration": 10.0
  }
}
```

### Multiple Email Recipients

```json
{
  "email": {
    "recipients": [
      "keeper1@zoo.com",
      "keeper2@zoo.com",
      "manager@zoo.com"
    ]
  }
}
```

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                           │
│                  (Orchestrator)                        │
└────┬──────────┬──────────┬──────────┬─────────────────┘
     │          │          │          │
     ▼          ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐
│Audio    │ │Feature │ │Model    │ │Activity  │
│Capture  │→│Extract │→│Inference│→│Tracker   │
└─────────┘ └────────┘ └─────────┘ └─────┬────┘
                                          │
     ┌────────────────────────────────────┘
     ▼
┌──────────┐    ┌────────────┐    ┌──────────┐
│Email     │    │RTC         │    │UI        │
│Sender    │    │Scheduler   │    │Controller│
└──────────┘    └────────────┘    └──────────┘
```

## Performance Characteristics

- **Latency**: < 2 seconds from audio to classification
- **Throughput**: 1-10 classifications per second (depending on vocalization frequency)
- **Power**: ~2.5W average, 4W peak
- **Battery Life**: 8+ hours with 20,000mAh power bank
- **Storage**: < 50 MB total (including OS and software)

## FAQ

**Q: Can I use a different microphone?**  
A: Yes, but must be I2S compatible. USB microphones require code changes to use PyAudio instead of I2S.

**Q: Does it work with other bat species?**  
A: Yes, if you retrain the model with your species data and replace the .tflite file.

**Q: Can I access it remotely?**  
A: Yes, enable SSH and connect via WiFi. Or use VNC for GUI access.

**Q: How long can it run autonomously?**  
A: Indefinitely with constant power. The RTC keeps time even if power is lost.

**Q: Can I change the email schedule?**  
A: Yes, use the rotary encoder UI or edit config.json directly.

## Support

For issues, check:
1. Hardware connections match wiring diagram
2. I2C and I2S enabled in raspi-config
3. All test_hardware.py checks pass
4. Logs in journalctl and bat_monitor.log

## License

This project uses the TensorFlow Lite model and bat vocalization classification system.
Hardware interface code is provided as-is for educational and research purposes.

---

**Created for bat conservation and welfare monitoring**  
Real-time analysis • No audio storage • Autonomous operation
